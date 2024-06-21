from src.runner.trainers.base_trainer import BaseTrainer
from tqdm import tqdm
import torch
import random
import logging
import numpy as np
import os
from src.utils.ifmorph_utils import get_grid, create_morphing


class WarpingTrainer(BaseTrainer):
    """The trainer for neural implicit warping.
    """
    def __init__(self, n_samples, warmup_steps, frame_dim, n_frames, fps, saved_dir, **kwargs):
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.warmup_steps = warmup_steps
        self.frame_dim = frame_dim
        self.n_frames = n_frames
        self.fps = fps
        self.saved_dir = saved_dir

    def train(self):
        if self.np_random_seeds is None:
            self.np_random_seeds = random.sample(range(10000000), k=self.num_epochs)

        while self.epoch <= self.num_epochs:
            # Reset the numpy random seed.
            np.random.seed(self.np_random_seeds[self.epoch - 1])

            # Do training and validation.
            print()
            logging.info(f'Epoch {self.epoch}.')
            train_log = self._run_epoch('training')
            logging.info(f'Train log: {train_log}.')

            if self.epoch % self.val_freq == 0:
                valid_log = self._run_epoch('validation')
                logging.info(f'Valid log: {valid_log}.')

            # Adjust the learning rate.
            if self.lr_scheduler is None:
                pass
            else:
                self.lr_scheduler.step()

            # Record the log information and visualization.
            if self.epoch % self.val_freq == 0:
                self.logger.write(self.epoch, train_log, None, None, valid_log, None, None, self.frame_dim)

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
            X, src_kpts, tgt_kpts = self._get_inputs_targets(batch)
            # yhat = self.net(X)
            # X = yhat["model_in"]  # requires_grad=True
            # X.allow_unused = True

            if mode == 'training':
                M = self.net(X)
                X = M['model_in'].squeeze()
                Y = M['model_out'].squeeze()
                X.allow_unused = True
            
                Ys = torch.cat((Y, -X[..., -1:]), dim=1)
                model_Xs = self.net(Ys)
                Xs = model_Xs['model_out']
                Yt = torch.cat((Y, 1 - X[..., -1:]), dim=1)
                model_Xt = self.net(Yt)
                Xt = model_Xt['model_out']
                Y1 = torch.cat((X[...,0:2], torch.ones_like(X[..., -1:])), dim=1)
                X1 = self.net(Y1)['model_out']

                losses = self._compute_losses(X, Y, Xs, Xt, X1, src_kpts, tgt_kpts)

                loss = (torch.stack(losses) * self.loss_weights).sum()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():

                    M = self.net(X)
                    X = M['model_in'].squeeze()
                    Y = M['model_out'].squeeze()
                    Ys = torch.cat((Y, -X[..., -1:]), dim=1)
                    model_Xs = self.net(Ys)
                    Xs = model_Xs['model_out'].squeeze()
                    Yt = torch.cat((Y, 1 - X[..., -1:]), dim=1)
                    model_Xt = self.net(Yt).squeeze()
                    Xt = model_Xt['model_out']
                    Y1 = torch.cat((X[...,0:2], torch.ones_like(X[..., -1:])), dim=1)
                    X1 = self.net(Y1)['model_out'].squeeze()

                    losses = self._compute_losses(X, Y, Xs, Xt, X1, src_kpts, tgt_kpts)

                    loss = (torch.stack(losses) * self.loss_weights).sum()
                    create_morphing(
                        warp_net=self.net,
                        frame0=self.valid_dataloader.dataset.initial_states[0],
                        frame1=self.valid_dataloader.dataset.initial_states[1],
                        output_path=os.path(self.saved_dir, f"epoch_{self.epoch}.mp4"),
                        frame_dims=self.frame_dim,
                        n_frames=self.n_frames,
                        fps=self.fps,
                        device=self.device,
                        landmark_src=src_kpts,
                        landmark_tgt=tgt_kpts,
                        plot_landmarks=True
                    )

            metrics =  self._compute_metrics(grid_input, src_kpts, tgt_kpts)

            batch_size = self.train_dataloader.batch_size if mode == 'training' else self.valid_dataloader.batch_size
            self._update_log(log, batch_size, loss, losses, metrics)
            count += batch_size
            trange.set_postfix(**dict((key, f'{value / count: .3f}') for key, value in log.items()))

        for key in log:
            log[key] /= count
        return log

    def _get_inputs_targets(self, batch):
        """Specify the data input and target.
        Args:
            batch (dict): A batch of data.
        Returns:
            input (torch.Tensor): The data input.
            target (torch.LongTensor): The data target.
        """
        return batch['grid_input'], batch['src_kpts'], batch['tgt_kpts']

    def _compute_losses(self, X, Y, Xs, Xt, X1, src_kpts, tgt_kpts):
        """Compute the losses.
        Args:
            output (torch.Tensor): The model output.
            target (torch.LongTensor): The data target.
        Returns:
            losses (list of torch.Tensor): The computed losses.
        """
        losses = []
        for loss in self.loss_fns:
            if loss.__class__.__name__ == 'TPSConstraintLoss':
                losses.append(loss(X, Y))
            elif loss.__class__.__name__ == 'InverseConstraintLoss':
                losses.append(loss(X, Xs, Xt, X1))
            elif loss.__class__.__name__ == 'IdentityConstraintLoss':
                losses.append(loss(X, Y))
            elif loss.__class__.__name__ == 'DataConstraintLoss':
                losses.append(loss(src_kpts, tgt_kpts, self.net))
            else:
                raise RuntimeError(f"Loss function {loss.__class__.__name__} cannot be recognized.")
        return losses

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
