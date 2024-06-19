import os.path as osp
import glob
import torch
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from src.utils.ifmorph_utils import get_grid
import math


class ImageDataset(Dataset):
    """A Simple image dataset.

    Parameters
    ----------
    path: str
        Path to the image, should be readable by PIL .

    sidelen: int, optional
        Grid width and height. If set to -1 (which is the default case),
        will infer it from the image size.

    channels_to_use: int, optional
        The number of channels of the input image to use during training. Must
        be 1 for grayscale or 3 for RGB images. Default is None, meaning that
        the all channels of the input image will be used, whether its 1 or 3.

    batch_size: int, optional, NOT USED
        Number of samples to fetch at each call to __getitem__. Default is
        None, meaning that the whole image will be returned. Useful for
        memory-constrained scenarios.

    Raises
    ------
    ValueError:
        If `len(sidelen)` > 2 or `len(sidelen)` == 0. Also if image does
        not have the correct mode. Accepted modes are: "RGB", "L" and "P".

    TypeError:
        If `type(sidelen)` not in [tuple, list, int]
    """
    def __init__(self, path: str, sidelen=-1, channels_to_use=None,
                 batch_size=None):
        super(ImageDataset, self).__init__()
        self.path = path
        if path[0] == "~":
            path = osp.expanduser(path)
        img = Image.open(path)

        grid_dims = None
        if sidelen != -1:
            if isinstance(sidelen, int):
                grid_dims = [sidelen] * 2
            elif isinstance(sidelen, (tuple, list)):
                if len(sidelen) > 2:
                    raise ValueError("sidelen has too many coordinates for"
                                     " image.")
                elif not sidelen:
                    raise ValueError("No grid size provided.")
                grid_dims = sidelen
            else:
                raise TypeError("sidelen is neither number or collection of"
                                " numbers.")
        else:
            grid_dims = (img.height, img.width)

        self.size = grid_dims

        if img.mode not in ["RGB", "L", "P"]:
            raise ValueError("Image must be RGB (3 channels) or grayscale"
                             f" (1 channel). # channels found: {len(img.mode)}"
                             f", format: {img.mode}")

        if channels_to_use is None:
            self.n_channels = 3 if img.mode == "RGB" else 1
        else:
            if channels_to_use not in [1, 3]:
                raise ValueError("Invalid number of channels to use. Should"
                                 f" be 1 or 3, found {channels_to_use}.")
            self.n_channels = channels_to_use

        t = [transforms.Resize(grid_dims), transforms.ToTensor()] if sidelen != -1 else [transforms.ToTensor()]
        if self.n_channels == 1:
            t.append(transforms.Grayscale())

        t = transforms.Compose(t)
        self.coords = get_grid(grid_dims)
        self.rgb = t(img).permute(1, 2, 0).view(-1, self.n_channels)
        if not batch_size:
            self.batch_size = self.rgb.shape[0]  # (H * W)
        else:
            self.batch_size = batch_size

    def pixels(self, coords=None):
        """
        Parameters
        ----------
        coords: torch.Tensor
            Point tensor in range [-1, 1]. Must have shape [N, 2].

        Returns
        -------
        rgb: torch.Tensor
            RGB values shaped [N, 3].
        """
        if coords is None:
            intcoords = (self.coords.detach().clone().cpu() * 0.5 + 0.5)
        else:
            intcoords = (coords.detach().clone().cpu() * 0.5 + 0.5)
        intcoords = intcoords.clamp(self.coords.min(), self.coords.max())
        intcoords[..., 0] *= self.size[0]
        intcoords[..., 1] *= self.size[1]
        intcoords = intcoords.floor().long()
        rgb = torch.zeros_like(self.rgb, device=self.coords.device)
        rgb = self.rgb[
            (intcoords[..., 0] * self.size[0]) + intcoords[..., 1],
            ...
        ]
        return rgb

    def __len__(self):
        # return math.ceil(self.rgb.shape[0] / self.batch_size)
        return 1

    def __getitem__(self, idx=None):
        """Returns the coordinates, RGB values and indices of pixels in image.

        Given a list of pixel indices `idx` returns their normalized
        coordinates, RGB values and their indices as well. If the list is not
        given (default), it will be generated and returned.

        Parameters
        ----------
        idx: list or torch.Tensor, optional
            Linearized pixel indices. If not given, will choose at random.

        Returns
        -------
        coords: torch.Tensor
            Nx2 linearized pixel coordinates

        rgb: torch.Tensor
            The Nx`self.n_channels` Pixel values. The number of columns depends
            on the number of channels of the input image.

        idx: torch.Tensor
            Indices of the pixels selected. If the `idx` parameter is provided,
            it will simply by a copy of it.
        """
        # if idx is None or not len(idx):
        #     iidx = torch.randint(self.coords.shape[0], (self.batch_size,))
        # elif not isinstance(idx, torch.Tensor):
        #     iidx = torch.Tensor(idx)
        # else:
        #     iidx = idx
        # iidx = iidx.to(self.coords.device)
        # # return self.coords[iidx, ...], self.rgb[iidx, ...], iidx

        iidx = torch.linspace(0, self.coords.shape[0]-1, self.batch_size, dtype=torch.long)
        out_dict = {
            'grid_coords': self.coords[iidx, ...],
            'rgb': self.rgb[iidx, ...],
            'idx': iidx
        }
        return out_dict 
