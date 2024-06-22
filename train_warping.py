import argparse
import logging
import os
import torch
import random
import yaml
import numpy as np
from box import Box
from pathlib import Path
from copy import deepcopy
import pandas as pd
from tqdm import tqdm

import src
from src.utils.ifmorph_utils import from_pth
from src.model.losses import WarpingLoss
from src.utils import create_morphing

def main(args):
    logging.info(f'Load the config from "{args.config}".')
    config = Box.from_yaml(filename=args.config)
    saved_dir = Path(config.main.saved_dir)
    if not saved_dir.is_dir():
        saved_dir.mkdir(parents=True)

    logging.info(f'Save the config to "{config.main.saved_dir}".')
    with open(saved_dir / 'config.yaml', 'w+') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)

    random.seed(config.main.random_seed)
    torch.manual_seed(random.getstate()[1][1])
    torch.cuda.manual_seed_all(random.getstate()[1][1])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logging.info('Create the device.')
    if 'cuda' in config.trainer.kwargs.device and not torch.cuda.is_available():
        raise ValueError("The cuda is not available. Please set the device in the trainer section to 'cpu'.")
    device = torch.device(config.trainer.kwargs.device)

    logging.info('Create the datasets.')
    # config.dataset.kwargs.update(initial_states=[im0, im1])
    dataset = _get_instance(src.datasets, config.dataset)

    logging.info('Load the neural images.')
    # im0 = from_pth(config.dataset.src_ckpt_path, w0=1, device=device)
    # im1 = from_pth(config.dataset.tgt_ckpt_path, w0=1, device=device)
    im0 = _get_instance(src.model.nets, config.im_net).to(device)
    im1 = _get_instance(src.model.nets, config.im_net).to(device)
    ckpt_im0 = torch.load(config.im_net.src_ckpt_path, map_location=device)
    ckpt_im1 = torch.load(config.im_net.tgt_ckpt_path, map_location=device)
    im0.load_state_dict(ckpt_im0['net'])
    im1.load_state_dict(ckpt_im1['net'])

    logging.info('Create the network architecture.')
    model = _get_instance(src.model.nets, config.net).to(device)

    logging.info('Create the loss functions and the corresponding weights.')
    src_kpts = np.load(config.dataset.src_kpts_path, allow_pickle=True)
    tgt_kpts = np.load(config.dataset.tgt_kpts_path, allow_pickle=True)
    src_kpts = torch.tensor(src_kpts).to(device)
    tgt_kpts = torch.tensor(tgt_kpts).to(device)
    best_loss = np.inf
    training_loss = {}
    loss_func = WarpingLoss(
        warp_src_pts=src_kpts,
        warp_tgt_pts=tgt_kpts,
        intermediate_times=config.losses.intermediate_times,
        constraint_weights=config.losses.constraint_weights,
    )

    logging.info('Create the optimizer.')
    optimizer = _get_instance(torch.optim, config.optimizer, model.parameters())

    logging.info('Start training.')
    training_steps = config.trainer.kwargs.num_epochs
    warmup_steps = config.trainer.kwargs.warmup_steps
    checkpoint_steps = config.trainer.kwargs.ckpt_step
    reconstruct_steps = config.trainer.kwargs.vis_step
    frame_dim = config.trainer.kwargs.frame_dim
    n_frames = config.trainer.kwargs.n_frames
    fps = config.trainer.kwargs.fps
    trange = tqdm(range(training_steps))
    log = {}
    log['Loss'] = 0
    for step in trange:
        X = dataset[0]
        X = X.to(device)


        # X = self.dataset.__getitem__()['grid_coords']
        #     print(X.mean(), X.std())
        #     X = X.detach().to(self.device).requires_grad_(False)
        #     rec = self.net(X, preserve_grad=True)["model_out"]
        #     print(rec.mean(), rec.std())
        #     rec = rec.detach().cpu().clip(0, 1).requires_grad_(False)

        # sz = [H, W, n_channels]
        # rec = rec.reshape(sz).permute((2, 0, 1))
        # img = to_pil_image(rec)
        # fname = self.dataset.path.split('/')[-1].split('.')[0]
        # out_img_path = osp.join(self.saved_dir, fname + "_recon.png")
        # img.save(out_img_path)
        # print(f"Image saved as: {out_img_path}")


        yhat = model(X)
        X = yhat["model_in"] 
        yhat = yhat["model_out"].squeeze()
        loss = loss_func(X, model)

        # Accumulating the losses.
        running_loss = torch.zeros((1, 1)).to(device)
        for k, v in loss.items():
            running_loss += v
            if k not in training_loss:
                training_loss[k] = [v.item()]
            else:
                training_loss[k].append(v.item())

        # if not step % 1000:
        #     print(step, running_loss.item())
        # Show training loss using trange.set_postfix
        # log = {k: v.item() for k, v in loss.items()}
        trange.set_postfix(**dict((key, f'{value: .3f}') for key, value in loss.items()))

        if step > warmup_steps and running_loss.item() < best_loss:
            best_step = step
            best_loss = running_loss.item()
            best_weights = deepcopy(model.state_dict())

        if checkpoint_steps is not None and step > 0 and not step % checkpoint_steps:
            torch.save(
                model.state_dict(),
                os.path.join(saved_dir, f"checkpoint_{step}.pth")
            )

        if reconstruct_steps is not None and step > 0 and not step % reconstruct_steps:
            logging.info(f"Create morphing videos at step {step}.")
            model = model.eval()
            vidpath = os.path.join(saved_dir, f"rec_{step}.mp4")
            with torch.no_grad():
                create_morphing(
                    warp_net=model,
                    frame0=im0,
                    frame1=im1,
                    output_path=vidpath,
                    frame_dims=frame_dim,
                    n_frames=n_frames,
                    fps=fps,
                    device=device,
                    landmark_src=src_kpts,
                    landmark_tgt=tgt_kpts,
                    plot_landmarks=False
                )
            model = model.train().to(device)

        optimizer.zero_grad()
        running_loss.backward()
        optimizer.step()

    logging.info("Training done.")
    logging.info(f"Best results at step {best_step}, with loss {best_loss}.")
    logging.info(f"Saving the results in folder {saved_dir}.")
    model = model.eval()
    with torch.no_grad():
        model.update_omegas(w0=1, ww=None)
        torch.save(
            model.state_dict(), os.path.join(saved_dir, "weights.pth")
        )

        model.w0 = config.net.kwargs.omega_0
        model.ww = config.net.kwargs.omega_w
        model.load_state_dict(best_weights)
        model.update_omegas(w0=1, ww=None)
        torch.save(
            model.state_dict(), os.path.join(saved_dir, "best.pth")
        )

    loss_df = pd.DataFrame.from_dict(training_loss)
    loss_df.to_csv(os.path.join(saved_dir, "loss.csv"), sep=";")

def _parse_args():
    parser = argparse.ArgumentParser(description="The script for the training and the testing.")
    parser.add_argument('--config', type=Path, help='The path of the config file.')
    args = parser.parse_args()
    return args


def _get_instance(module, config, *args):
    """
    Args:
        module (module): The python module.
        config (Box): The config to create the class object.

    Returns:
        instance (object): The class object defined in the module.
    """
    cls = getattr(module, config.name)
    kwargs = config.get('kwargs')
    return cls(*args, **config.kwargs) if kwargs else cls(*args)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_args()
    main(args)
