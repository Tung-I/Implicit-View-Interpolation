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

    # logging.info(f'Save the config to "{config.main.saved_dir}".')
    # with open(saved_dir / 'config.yaml', 'w+') as f:
    #     yaml.dump(config.to_dict(), f, default_flow_style=False)

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

    config.dataset.name = 'ImageDataset'
    config.dataset.kwargs = {
        'path': config.dataset.src_img_path,
        'sidelen': config.trainer.kwargs.frame_dim,
    }
    dataset0 = _get_instance(src.datasets, config.dataset)

    config.dataset.name = 'ImageDataset'
    config.dataset.kwargs = {
        'path': config.dataset.tgt_img_path,
        'sidelen': config.trainer.kwargs.frame_dim,
    }
    dataset1 = _get_instance(src.datasets, config.dataset)

    # if not args.cont:
    #     logging.info('Create the datasets.')
    #     dataset0 = _get_instance(src.datasets, config.dataset_0)
    #     dataset1 = _get_instance(src.datasets, config.dataset_1)
    # else:
    #     logging.info('Load the neural images.')
    #     # im0 = from_pth(config.dataset.src_ckpt_path, w0=1, device=device)
    #     # im1 = from_pth(config.dataset.tgt_ckpt_path, w0=1, device=device)
    #     im0 = _get_instance(src.model.nets, config.im_net).to(device)
    #     im1 = _get_instance(src.model.nets, config.im_net).to(device)
    #     ckpt_im0 = torch.load(config.im_net.src_ckpt_path, map_location=device)
    #     ckpt_im1 = torch.load(config.im_net.tgt_ckpt_path, map_location=device)
    #     im0.load_state_dict(ckpt_im0['net'])
    #     im1.load_state_dict(ckpt_im1['net'])
    #     im0.eval()
    #     im1.eval()

    logging.info('Create the network architecture.')
    model = _get_instance(src.model.nets, config.net).to(device)
    ckpt_path = os.path.join(config.main.saved_dir, 'checkpoint_8000.pth')
    # ckpt_path = os.path.join(config.main.saved_dir, 'checkpoint_4000.pth')
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()


    logging.info('Start inferencing.')
    training_steps = config.trainer.kwargs.num_epochs
    frame_dim = config.trainer.kwargs.frame_dim
    n_frames = config.trainer.kwargs.n_frames
    fps = config.trainer.kwargs.fps

    with torch.no_grad():
        vidpath = os.path.join(saved_dir, f"recon.mp4")
        create_morphing(
            warp_net=model,
            frame0=dataset0,
            frame1=dataset1,
            output_path=vidpath,
            frame_dims=frame_dim,
            n_frames=n_frames,
            fps=fps,
            device=device,
            landmark_src=None,
            landmark_tgt=None,
            plot_landmarks=False,
            continuous=False
        )

    # with torch.no_grad():
    #     if args.cont:
    #         vidpath = os.path.join(saved_dir, f"rec_neural_image.mp4")
    #         create_morphing(
    #             warp_net=model,
    #             frame0=im0,
    #             frame1=im1,
    #             output_path=vidpath,
    #             frame_dims=frame_dim,
    #             n_frames=n_frames,
    #             fps=fps,
    #             device=device,
    #             landmark_src=None,
    #             landmark_tgt=None,
    #             plot_landmarks=False, 
    #             continuous=True
    #         )
    #     else:
    #         vidpath = os.path.join(saved_dir, f"rec_noncont.mp4")
    #         create_morphing(
    #             warp_net=model,
    #             frame0=dataset0,
    #             frame1=dataset1,
    #             output_path=vidpath,
    #             frame_dims=frame_dim,
    #             n_frames=n_frames,
    #             fps=fps,
    #             device=device,
    #             landmark_src=None,
    #             landmark_tgt=None,
    #             plot_landmarks=False,
    #             continuous=False
    #         )


def _parse_args():
    parser = argparse.ArgumentParser(description="The script for the training and the testing.")
    parser.add_argument('--config', type=Path, help='The path of the config file.')
    parser.add_argument('--cont', action='store_true', help='Whether to use continuous SIREN representation.')
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
