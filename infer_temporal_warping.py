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
    dataset0 = _get_instance(src.datasets, config.dataset_0)
    dataset1 = _get_instance(src.datasets, config.dataset_1)
   

    logging.info('Create the network architecture.')
    model = _get_instance(src.model.nets, config.net).to(device)
    ckpt = torch.load(config.net.ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()


    logging.info('Start inferencing.')
    training_steps = config.trainer.kwargs.num_epochs
    frame_dim = config.trainer.kwargs.frame_dim
    n_frames = config.trainer.kwargs.n_frames
    fps = config.trainer.kwargs.fps
    with torch.no_grad():
        vidpath = os.path.join(saved_dir, f"rec_noncont.mp4")
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
