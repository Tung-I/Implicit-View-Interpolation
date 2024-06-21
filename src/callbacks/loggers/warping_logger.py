import torch
import random
import numpy as np
from torchvision.utils import make_grid

from .base_logger import BaseLogger

class WarpingLogger(BaseLogger):
    """Logger for image warping tasks.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_images(self, epoch, train_batch, train_output, valid_batch, valid_output, im_dim):
        """
        Args:
            epoch (int): The number of trained epochs.
            train_batch (dict): The training batch.
            train_output (dict): The training output.
            valid_batch (dict): The validation batch.
            valid_output (dict): The validation output.
            im_dim (tuple): The dimensions of the images (H, W).
        """
        # H, W = im_dim

        # # Reshape and prepare training images
        # train_output_img = train_output.reshape(-1, H, W, 3).permute(0, 3, 1, 2)
        # train_target_img = train_batch['rgb'].reshape(-1, H, W, 3).permute(0, 3, 1, 2)

        # # Reshape and prepare validation images
        # valid_output_img = valid_output.reshape(-1, H, W, 3).permute(0, 3, 1, 2)
        # valid_target_img = valid_batch['rgb'].reshape(-1, H, W, 3).permute(0, 3, 1, 2)

        # # Create image grids
        # train_pred_grid = make_grid(train_output_img, nrow=1, normalize=True, scale_each=True, pad_value=1)
        # train_target_grid = make_grid(train_target_img, nrow=1, normalize=True, scale_each=True, pad_value=1)
        # train_combined = torch.cat((train_target_grid, train_pred_grid), dim=-1)

        # valid_pred_grid = make_grid(valid_output_img, nrow=1, normalize=True, scale_each=True, pad_value=1)
        # valid_target_grid = make_grid(valid_target_img, nrow=1, normalize=True, scale_each=True, pad_value=1)
        # valid_combined = torch.cat((valid_target_grid, valid_pred_grid), dim=-1)

        # # Log images to tensorboard
        # self.writer.add_image('train', train_combined, epoch)
        # self.writer.add_image('valid', valid_combined, epoch)
        return
