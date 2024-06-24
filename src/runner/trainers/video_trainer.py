from src.runner.trainers.base_trainer import BaseTrainer
from tqdm import tqdm
import torch
import random
import logging
import numpy as np


class VideoTrainer(BaseTrainer):
    def __init__(self, frame_dims, iframe_net, **kwargs):
        super().__init__(**kwargs)
        self.frame_dims = frame_dims
        self.iframe_net = iframe_net.to(self.device)
        # self.res_net = res_net.to(self.device)
        # self.res_start_epoch = res_start_epoch
        # self.join_train_epoch = join_train_epoch

    def train(self):
        if self.np_random_seeds is None:
            self.np_random_seeds = random.sample(range(10000000), k=self.num_epochs)

        while self.epoch <= self.num_epochs:
            # Reset the numpy random seed.
            np.random.seed(self.np_random_seeds[self.epoch - 1])

            # # Unfreeze the res_net parameters
            # if self.epoch == self.res_start_epoch:
            #     for param in self.res_net.parameters():
            #         param.requires_grad = True
            #     for param in self.net.parameters():  # Freeze the temp_war_net parameters
            #         param.requires_grad = False
            
            # if self.epoch == self.join_train_epoch:
            #     for param in self.net.parameters():
            #         param.requires_grad = True

            # Do training and validation.
            print()
            logging.info(f'Epoch {self.epoch}.')
            train_log, train_batch, train_outputs = self._run_epoch('training')
            logging.info(f'Train log: {train_log}.')

            if self.epoch % self.val_freq == 0:
                valid_log, valid_batch, valid_outputs = self._run_epoch('validation')
                logging.info(f'Valid log: {valid_log}.')

            # Adjust the learning rate.
            if self.lr_scheduler is None:
                pass
            else:
                self.lr_scheduler.step()

            # Record the log information and visualization.
            if self.epoch % self.val_freq == 0:
                # self.logger.write(self.epoch, train_log, train_batch, train_outputs.unsqueeze(0),
                #                 valid_log, valid_batch, valid_outputs.unsqueeze(0), self.im_dim)
                self.logger.write(self.epoch, train_log, train_batch, train_outputs,
                                valid_log, valid_batch, valid_outputs, self.frame_dims)

            # Save the regular checkpoint.
            saved_path = self.monitor.is_saved(self.epoch)
            if saved_path:
                logging.info(f'Save the checkpoint to {saved_path}.')
                self.save(saved_path)

            # Save the best checkpoint.
            if self.epoch % self.val_freq == 0:
                saved_path = self.monitor.is_best(valid_log)
                if saved_path:
                    logging.info(f'Save the best checkpoint to {saved_path} ({self.monitor.mode} {self.monitor.target}: {self.monitor.best}).')
                    self.save(saved_path)
                else:
                    logging.info(f'The best checkpoint is remained (at epoch {self.epoch - self.monitor.not_improved_count}, {self.monitor.mode} {self.monitor.target}: {self.monitor.best}).')

            # Early stop.
            if self.monitor.is_early_stopped():
                logging.info('Early stopped.')
                break

            self.epoch +=1

        self.logger.close()

    def _run_epoch(self, mode):
        if mode == 'training':
            self.net.train()
        else:
            self.net.eval()
        dataloader = self.train_dataloader if mode == 'training' else self.valid_dataloader
        trange = tqdm(dataloader,
                      total=len(dataloader),
                      desc=mode)

        log = self._init_log()
        count = 0
        for batch in trange:
            batch = self._allocate_data(batch)
            coords, rgb, _, t = self._get_inputs_targets(batch)
            H, W = self.frame_dims

            if mode == 'training':
                outputs, losses = self._compute_losses(coords, rgb, t)
                loss = (torch.stack(losses) * self.loss_weights).sum()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    outputs, losses = self._compute_losses(coords, rgb, t)
                    loss = (torch.stack(losses) * self.loss_weights).sum()
                    # ##############
                    # outputs = self.iframe_net(coords, preserve_grad=True)["model_out"]
                    # ###############

            H, W = self.frame_dims
            metrics =  self._compute_metrics(outputs.view(-1, H, W, 3).permute(0, 3, 2, 1), rgb.view(-1, H, W, 3).permute(0, 3, 2, 1))
            # metrics = self._compute_metrics(outputs, rgb)

            batch_size = self.train_dataloader.batch_size if mode == 'training' else self.valid_dataloader.batch_size
            self._update_log(log, batch_size, loss, losses, metrics)
            count += batch_size
            trange.set_postfix(**dict((key, f'{value / count: .3f}') for key, value in log.items()))

        for key in log:
            log[key] /= count

        # Clamp output values to [0, 1]
        outputs = outputs.clamp(0, 1)

        return log, batch, outputs

    def _get_inputs_targets(self, batch):
        """Specify the data input and target.
        Args:
            batch (dict): A batch of data.
        Returns:
            input (torch.Tensor): The data input.
            target (torch.LongTensor): The data target.
        """
        return batch['grid_coords'].detach().to(self.device).requires_grad_(False),\
              batch['rgb'].detach().to(self.device).requires_grad_(False),\
                  batch['idx'].detach().to(self.device).requires_grad_(False), \
                    batch['time_step']

    def _compute_losses(self, coords, rgb, t):
        """Compute the losses.
        Args:
            output (torch.Tensor): The model output.
            target (torch.LongTensor): The data target.
        Returns:
            losses (list of torch.Tensor): The computed losses.
        """
        # temporal warping
        coords = coords.squeeze() # (20000, 1)
        time_coord = torch.ones_like(coords[..., :1], dtype=torch.float32) * t  # (20000, 1)
        coords_time = torch.cat((coords, time_coord), dim=1).to(self.device)
        delta_coords = self.net(coords_time)["model_out"].squeeze() # (N, 2)

        # get the rgb values at time t
        outputs = self.iframe_net(coords + delta_coords, preserve_grad=True)["model_out"]

        # if self.epoch >= self.res_start_epoch:
        #     res_outputs = self.res_net(coords_time)["model_out"].squeeze()
        #     outputs = outputs + res_outputs

        H, W = self.frame_dims
        losses = [loss(outputs.view(-1, H, W, 3).permute(0, 3, 2, 1), rgb.view(-1, H, W, 3).permute(0, 3, 2, 1)) for loss in self.loss_fns]
        return outputs, losses

    def _compute_metrics(self, output, target):
        """Compute the metrics.
        Args:
             output (torch.Tensor): The model output.
             target (torch.LongTensor): The data target.
        Returns:
            metrics (list of torch.Tensor): The computed metrics.
        """
        metrics = [metric(output, target) for metric in self.metric_fns]
        return metrics

    def _init_log(self):
        """Initialize the log.
        Returns:
            log (dict): The initialized log.
        """
        log = {}
        log['Loss'] = 0
        for loss in self.loss_fns:
            log[loss.__class__.__name__] = 0
        for metric in self.metric_fns:
            log[metric.__class__.__name__] = 0
        return log

    def _update_log(self, log, batch_size, loss, losses, metrics):
        """Update the log.
        Args:
            log (dict): The log to be updated.
            batch_size (int): The batch size.
            loss (torch.Tensor): The weighted sum of the computed losses.
            losses (list of torch.Tensor): The computed losses.
            metrics (list of torch.Tensor): The computed metrics.
        """
        log['Loss'] += loss.item() * batch_size
        for loss, _loss in zip(self.loss_fns, losses):
            log[loss.__class__.__name__] += _loss.item() * batch_size
        for metric, _metric in zip(self.metric_fns, metrics):
            log[metric.__class__.__name__] += _metric.item() * batch_size

    def save(self, path):
        """Save the model checkpoint.
        Args:
            path (Path): The path to save the model checkpoint.
        """
        torch.save({
            'net': self.net.state_dict(),
            # 'res_net': self.res_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'monitor': self.monitor,
            'epoch': self.epoch,
            'random_state': random.getstate(),
            'np_random_seeds': self.np_random_seeds
        }, path)
