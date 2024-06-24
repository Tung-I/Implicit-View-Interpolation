import argparse
import logging
import os
import torch
import random
import yaml
from box import Box
from pathlib import Path

import src

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
    if 'cuda' in config.predictor.kwargs.device and not torch.cuda.is_available():
        raise ValueError("The cuda is not available. Please set the device in the trainer section to 'cpu'.")
    device = torch.device(config.predictor.kwargs.device)

    logging.info('Create the training and validation datasets.')
    config.dataset.kwargs.update(_type='train')
    train_dataset = _get_instance(src.datasets, config.dataset)

    logging.info('Create the dataloaders.')
    cls = getattr(src.datasets, config.dataset.name)
    train_batch_size = config.dataloader.kwargs.pop('train_batch_size')
    config.dataloader.kwargs.update(collate_fn=getattr(cls, 'collate_fn', None), batch_size=train_batch_size)
    train_dataloader = _get_instance(src.datasets, config.dataloader, train_dataset)

    logging.info('Create the temporal warping network architecture.')
    temp_war_net = _get_instance(src.model.nets, config.net)
    ckpt = torch.load(config.net.ckpt_path, map_location=device)
    temp_war_net.load_state_dict(ckpt['net'])
    temp_war_net.eval()

    logging.info('Create the I-frame network architecture.')
    iframe_net = _get_instance(src.model.nets, config.iframe_net)
    iframe_ckpt = torch.load(config.iframe_net.ckpt_path, map_location=device)
    iframe_net.load_state_dict(iframe_ckpt['net'])
    iframe_net.eval()

    # logging.info('Create the residual color network architecture.')
    # res_net = _get_instance(src.model.nets, config.res_net)
    # res_ckpt = torch.load(config.res_net.ckpt_path, map_location=device)
    # res_net.load_state_dict(res_ckpt['res_net'])
    # res_net.eval()

    logging.info('Create the predictor.')
    kwargs = {'device': device,
                'dataloader': train_dataloader,
                'net': temp_war_net,
                'iframe_net': iframe_net,
                # 'res_net': res_net,
                'frame_dims': config.dataset.kwargs.frame_dims, 
                'saved_dir': saved_dir}
    config.predictor.kwargs.update(kwargs)
    predictor= _get_instance(src.runner.predictors, config.predictor)


    predictor.predict()
    logging.info('End inference.')

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
