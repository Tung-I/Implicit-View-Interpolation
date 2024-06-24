import yaml
import argparse
import os

VIDEO_SEGMENT_LENGTH = 15

def generate_neural_image_yaml(cfg_out_dir, model_saved_dir, data_dir, seg_idx, output_file):

    config = {
        'main': {
            'random_seed': '6666',
            'saved_dir': os.path.join(model_saved_dir)
        },
        'dataset': {
            'name': 'ImageDataset',
            'kwargs': {
                'path': os.path.join(data_dir, '{0:03d}.png'.format(seg_idx*VIDEO_SEGMENT_LENGTH)),
                'sidelen': [270, 480]
            }
        },
        'dataloader': {
            'name': 'Dataloader',
            'kwargs': {
                'train_batch_size': 1,
                'valid_batch_size': 1,
                'shuffle': False,
                'num_workers': 0
            }
        },
        'net': {
            'name': 'SIREN',
            'kwargs': {
                'in_channels': 2,
                'out_channels': 3,
                'hidden_layer_config': [256, 256, 256],
                'omega_0': 100,
                'omega_w': 100,
                'delay_init': False
            }
        },
        'losses': [
            {'name': 'MSELoss', 'weight': 1.0},
            {'name': 'L1Loss', 'weight': 0.1}
        ],
        'metrics': [
            {'name': 'PSNR'}
        ],
        'optimizer': {
            'name': 'Adam',
            'kwargs': {
                'lr': 0.0001
            }
        },
        'lr_scheduler': {
            'name': 'StepLR',
            'kwargs': {
                'step_size': 1200,
                'gamma': 0.1
            }
        },
        'logger': {
            'name': 'ImageLogger',
            'kwargs': {
                'dummy_input': False
            }
        },
        'monitor': {
            'name': 'Monitor',
            'kwargs': {
                'mode': 'min',
                'target': 'Loss',
                'saved_freq': 10000000,
                'early_stop': 0
            }
        },
        'trainer': {
            'name': 'ImageTrainer',
            'kwargs': {
                'device': 'cuda:0',
                'num_epochs': 1200,
                'val_freq': 50
            }
        }
    }

    os.makedirs(cfg_out_dir, exist_ok=True)
    yaml_path = os.path.join(cfg_out_dir, output_file)
    with open(yaml_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)



def generate_neural_video_yaml(cfg_out_dir, model_saved_dir, output_file, data_dir, ckpt_path, seg_idx):
    config = {
        'main': {
            'random_seed': '6666',
            'saved_dir': model_saved_dir
        },
        'dataset': {
            'name': 'VideoDataset',
            'kwargs': {
                'data_dir': data_dir,
                'frame_dims': [270, 480],
                'frame_range': [seg_idx*VIDEO_SEGMENT_LENGTH, seg_idx*VIDEO_SEGMENT_LENGTH+VIDEO_SEGMENT_LENGTH-1],
            }
        },
        'dataloader': {
            'name': 'Dataloader',
            'kwargs': {
                'train_batch_size': 1,
                'valid_batch_size': 1,
                'shuffle': True,
                'num_workers': 0
            }
        },
        'iframe_net': {
            'ckpt_path': ckpt_path,
            'name': 'SIREN',
            'kwargs': {
                'in_channels': 2,
                'out_channels': 3,
                'hidden_layer_config': [256, 256, 256],
                'omega_0': 100,
                'omega_w': 100,
                'delay_init': False
            }
        },
        'net': {
            'name': 'SIREN',
            'kwargs': {
                'in_channels': 3,
                'out_channels': 2,
                'hidden_layer_config': [256, 256, 256],
                'omega_0': 24,
                'omega_w': 24,
                'delay_init': False
            }
        },
        'losses': [
            {'name': 'MSELoss', 'weight': 1.0},
            {'name': 'L1Loss', 'weight': 0.1}
        ],
        'metrics': [
            {'name': 'PSNR'}
        ],
        'optimizer': {
            'name': 'Adam',
            'kwargs': {
                'lr': 0.0001
            }
        },
        'lr_scheduler': {
            'name': 'StepLR',
            'kwargs': {
                'step_size': 200,
                'gamma': 0.1
            }
        },
        'logger': {
            'name': 'ImageLogger',
            'kwargs': {
                'dummy_input': False
            }
        },
        'monitor': {
            'name': 'Monitor',
            'kwargs': {
                'mode': 'min',
                'target': 'Loss',
                'saved_freq': 10000000,
                'early_stop': 0
            }
        },
        'trainer': {
            'name': 'VideoTrainer',
            'kwargs': {
                'device': 'cuda:0',
                'num_epochs': 320,
                'val_freq': 20,
            }
        }
    }

    os.makedirs(cfg_out_dir, exist_ok=True)
    yaml_path = os.path.join(cfg_out_dir, output_file)
    with open(yaml_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate YAML configuration file for experiments.')
    parser.add_argument('--cfg_out_dir', type=str, help='The output directory to save the YAML file.')
    parser.add_argument('--model_saved_dir', type=str, help='The output directory to save the model.')
    parser.add_argument('--output_file', type=str, help='The name of the output YAML file.')
    parser.add_argument('--data_dir', type=str, help='The data directory for the dataset.')
    parser.add_argument('--ckpt_path', type=str, help='The checkpoint path for iframe_net.')
    parser.add_argument('--seg_idx', type=int, help='The segment index to generate the YAML file for.')
    parser.add_argument('--task', type=str, help='The task to generate the YAML file for.')

    args = parser.parse_args()

    if args.task == 'neural_video':
        generate_neural_video_yaml(args.cfg_out_dir, args.model_saved_dir, args.output_file, args.data_dir, args.ckpt_path, args.seg_idx)
    elif args.task == 'neural_image':
        generate_neural_image_yaml(args.cfg_out_dir, args.model_saved_dir, args.data_dir, args.seg_idx, args.output_file)
    else:
        raise ValueError(f'Invalid task: {args.task}.')