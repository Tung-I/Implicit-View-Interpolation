import yaml
import argparse
import os
import os.path as osp

def generate_neural_image_yaml(cfg_out_dir, model_saved_dir, im_path):
    config = {
        'main': {
            'random_seed': '6666',
            'saved_dir': os.path.join(model_saved_dir, im_path.split('/')[-1].split('.')[0])
        },
        'dataset': {
            'name': 'ImageDataset',
            'kwargs': {
                'path': im_path,
                # 'sidelen': [2028, 2704]
                'sidelen': [1014, 1352]
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
                'step_size': 900,
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
                'val_freq': 100
            }
        }
    }

    os.makedirs(cfg_out_dir, exist_ok=True)
    yaml_fname = im_path.split('/')[-1].split('.')[0] + '.yaml'
    out_path = os.path.join(cfg_out_dir, yaml_fname)
    print(f'Saving YAML file to {out_path}')
    with open(out_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

def generate_neural_warping_yaml(data_dir, src_id, tgt_id, n_kpts):
    """
    data_dir: data/banjoman
    src_id: 000
    tgt_id: 001
    """
    pair_dir = osp.join(data_dir, f'cam{src_id}_cam{tgt_id}')
    out_dir = osp.join(data_dir, f'cam{src_id}_cam{tgt_id}', f'{n_kpts}')
    config = {
        'main': {
            'random_seed': '6666',
            'saved_dir': out_dir
        },
        'dataset': {
            'name': 'WarpingDataset',
            'src_kpts_path': osp.join(pair_dir, f'{n_kpts}', f"cam{src_id}.dat"),
            'tgt_kpts_path': osp.join(pair_dir, f'{n_kpts}', f"cam{tgt_id}.dat"),
            'src_img_path': osp.join(data_dir, f'cam{src_id}.png'),
            'tgt_img_path': osp.join(data_dir, f'cam{tgt_id}.png'),
            'kwargs': {
                'time_range': [-1.0, 1.0],
                'n_samples': 20000
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
        'im_net': {
            'src_ckpt_path': osp.join(data_dir, 'image_models', f'cam{src_id}', 'checkpoints', 'model_best.pth'),
            'tgt_ckpt_path': osp.join(data_dir, 'image_models', f'cam{tgt_id}', 'checkpoints', 'model_best.pth'),
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
                'hidden_layer_config': [128, 128],
                'omega_0': 24,
                'omega_w': 24,
                'delay_init': False
            }
        },
        'losses': {
            'intermediate_times': [0.16, 0.32, 0.5, 0.66, 0.82],
            'constraint_weights': {
                'data_constraint': 10000,
                'identity_constraint': 1000,
                'inv_constraint': 10000,
                'TPS_constraint': 1000
            }
        },
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
                'step_size': 7000,
                'gamma': 0.1
            }
        },
        'logger': {
            'name': 'WarpingLogger',
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
            'name': 'WarpingTrainer',
            'kwargs': {
                'device': 'cuda:0',
                'num_epochs': 8001,
                'ckpt_step': 4000,
                'vis_step': 40000000000,
                'n_samples': 20000,
                'warmup_steps': 1000,
                'frame_dim': [2028, 2704],
                # 'frame_dim': [1014, 1302],
                'n_frames': 101,
                'fps': 20
            }
        }
    }

    yaml_fname = f'{src_id}_{tgt_id}.yaml'
    out_path = os.path.join(out_dir, yaml_fname)
    print(f'Saving YAML file to {out_path}')
    with open(out_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate YAML configuration file for experiments.')
    parser.add_argument('--cfg_dir', type=str, help='The output directory to save the YAML file.')
    parser.add_argument('--model_dir', type=str, help='The output directory to save the model.')
    parser.add_argument('--im_path', type=str, help='The path to the image file.')
    parser.add_argument('--task', type=str, help='The task to generate the YAML file for.')
    parser.add_argument('--im1', type=str, help='The path to the first image file.')
    parser.add_argument('--im2', type=str, help='The path to the first image file.')
    parser.add_argument('--kpts', type=int, default=2048, help='The number of keypoints to extract.')

    args = parser.parse_args()

    if args.task == 'image':
        generate_neural_image_yaml(args.cfg_dir, args.model_dir, args.im_path)
    elif args.task == 'warping':
        generate_neural_warping_yaml(args.cfg_dir, args.im1, args.im2, args.kpts)
    else:
        raise ValueError(f'Invalid task: {args.task}.')