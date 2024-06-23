import argparse
import logging
import yaml
import torch
import random
import numpy as np
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

    logging.info('Create the device.')
    if 'cuda' in config.predictor.kwargs.device and not torch.cuda.is_available():
        raise ValueError("The cuda is not available. Please set the device in the predictor section to 'cpu'.")
    device = torch.device(config.predictor.kwargs.device)

    logging.info('Create the training and validation datasets.')
    pred_dataset = _get_instance(src.datasets, config.dataset)

    logging.info(f'Load the previous checkpoint from "{config.main.loaded_path}".')
    net = _get_instance(src.model.nets, config.net).to(device)
    checkpoint = torch.load(config.main.loaded_path, map_location=device)
    net.load_state_dict(checkpoint['net'])
    net.eval()
    # net.load_state_dict(checkpoint)
    
    logging.info('Create the metric functions.')
    metric_fns = [_get_instance(src.model, config_metric) for config_metric in config.metrics]

    logging.info('Create the predictor.')
    config.predictor.kwargs.update(net=net, metric_fns=metric_fns, device=device,
                                   saved_dir=saved_dir, dataset=pred_dataset)
    predictor = _get_instance(src.runner.predictors, config.predictor)
    
    logging.info('Start testing.')
    predictor.predict()
    logging.info('End testing.')


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