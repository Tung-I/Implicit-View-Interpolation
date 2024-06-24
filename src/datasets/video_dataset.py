import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from src.utils.ifmorph_utils import get_grid
from PIL import Image

class VideoDataset(Dataset):
    def __init__(self, data_dir: str, frame_dims=[540, 960], batch_size=None, frame_range=None, _type='train'):
        super(VideoDataset, self).__init__()
        self.data_dir = os.path.expanduser(data_dir) if data_dir[0] == "~" else data_dir
        self.frame_paths = sorted([os.path.join(self.data_dir, fname) for fname in os.listdir(self.data_dir) if fname.endswith('.png')])
        if not self.frame_paths:
            raise ValueError(f"No images found in directory {data_dir}")
        
        if frame_range is not None:
            self.frame_paths = self.frame_paths[frame_range[0]:frame_range[1]+1]
        self.frame_idxs = list(range(len(self.frame_paths)))

        self.frame_dims = frame_dims
        self.n_channels = 3 

        t = [transforms.ToTensor()]
        self.transform = transforms.Compose(t)
        self.coords = get_grid(self.frame_dims)
        if not batch_size:
            self.batch_size = self.coords.shape[0]  # (H * W)
        else:
            self.batch_size = batch_size

        self.type = _type
        # self.n_pixel_samples = n_pixel_samples

        self.rgbs = []
        for frame_path in self.frame_paths:
            img = Image.open(frame_path)
            rgb = self.transform(img).permute(1, 2, 0).view(-1, self.n_channels)
            self.rgbs.append(rgb)

    # def pixels(self, img, coords=None):
    #     t_img = self.transform(img).permute(1, 2, 0).view(-1, self.n_channels)
    #     if coords is None:
    #         intcoords = (self.coords.detach().clone().cpu() * 0.5 + 0.5)
    #     else:
    #         intcoords = (coords.detach().clone().cpu() * 0.5 + 0.5)
    #     intcoords = intcoords.clamp(self.coords.min(), self.coords.max())
    #     intcoords[..., 0] *= self.size[0]
    #     intcoords[..., 1] *= self.size[1]
    #     intcoords = intcoords.floor().long()
    #     rgb = torch.zeros_like(t_img, device=self.coords.device)
    #     rgb = t_img[
    #         (intcoords[..., 0] * self.size[0]) + intcoords[..., 1],
    #         ...
    #     ]
    #     return rgb

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        """Returns the coordinates, RGB values and indices of pixels in the specified frame.

        Given an image index `idx`, loads the image and returns its normalized
        coordinates, RGB values, and the indices of the pixels.

        Parameters
        ----------
        idx: int
            Index of the image/frame to be loaded.

        Returns
        -------
        out_dict: dict
            Dictionary containing 'grid_coords', 'rgb', and 'idx'.
        """
      
        frame_idx = self.frame_idxs[idx]
        rgb = self.rgbs[frame_idx]

        iidx = torch.linspace(0, self.coords.shape[0]-1, self.batch_size, dtype=torch.long)
        normalized_time = torch.tensor(float(frame_idx) / len(self.frame_idxs), dtype=torch.float32)
        
        out_dict = {
            'grid_coords': self.coords[iidx, ...],
            'rgb': rgb[iidx, ...],
            'idx': iidx,
            'time_step': normalized_time  # Normalized time step between 0 and 1
        }
        return out_dict