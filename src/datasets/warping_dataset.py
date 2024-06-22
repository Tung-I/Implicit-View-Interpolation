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

class WarpingDataset(Dataset):
    """Warping dataset.

    Parameters
    ----------
    initial_states: list[tuples[number, torch.nn.Module]]
        A list of initial states (known images), and the time of each image.
        Note that the time should be in range [-1, 1]. And we will only sample
        in the time-range given here, i.e., if this parameter is
        [(-0.5, ...), (0.8, ...)], we will only sample values in range
        [-0.5, 0.8].

    num_samples: int
        Number of samples to draw at every call to __getitem__. Note that half
        of the samples will be drawn at the initial states (evenly distributed
        between them), and the other half will be drawn at intermediate times.

    device: str, torch.Device, optional
        The device to store the read models. By default is cpu.

    grid_sampling: boolean, optional
        Set to `True` (default) to sample points in a grid, distributed
        uniformely along time, or `False` to randomly sample points in the
        [-1, 1] domain.

    Examples
    --------
    > # Creating a dataset with 3 initial states at times -0.8, 0.4 and 0.95,
    > # all on CPU. We will fetch 1000 points per call to __getitem__.
    > initial_states = [("m1.pth", -0.8), ("m2.pth", 0.4), "(m3.pth", 0.95)]
    > data = WarpingDataset(initial_states, 1000, torch.device("cpu"))
    > X = data[0]
    > print(X.shape)  # Should print something like: [1000, 3]
    """
    # def __init__(self, initial_states: list[torch.nn.Module, torch.nn.Module], kpts_pair: list[np.array, np.array],
    #              n_samples: int, known_times=[0., 1.0], time_range=[-1.0, 1.0], device: str = "cuda:0", grid_sampling: bool = True):
    def __init__(self,
                 n_samples: int, known_times=[0., 1.0], time_range=[-1.0, 1.0], device: str = "cuda:0", grid_sampling: bool = True):
        super(WarpingDataset, self).__init__()
        self.num_samples = n_samples
        self.device = device
        self.grid_sampling = grid_sampling

        # self.initial_states = initial_states
        self.known_times = known_times

        self.time_range = time_range

        # Spatial coordinates
        if self.grid_sampling:
            N = self.num_samples // 2
            m = int(math.sqrt(N))
            self.coords = get_grid([m, m]).to(self.device)
            self.int_times = 2 * (torch.arange(0, N, 1, device=self.device, dtype=torch.float32) - (N / 2)) / N

        # # Keypoints of the matching
        # self.src_kpts = torch.from_numpy(kpts_pair[0]).float()
        # self.tgt_kpts = torch.from_numpy(kpts_pair[1]).float()

    # @property
    # def initial_conditions(self):
    #     return list(zip(self.initial_states, self.known_times))

    def __len__(self):
        return 1

    def __getitem__(self, _):
        """
        Returns
        -------
        X: torch.Tensor
            A [`num_samples`, 3] shaped tensor with the pixel coordinates at
            the first two columns, and time coordinates at the last column.
        """
        # # Spatial coordinates
        N = self.num_samples // 2

        if self.grid_sampling:
            int_times = self.int_times[torch.randperm(N, device=self.device)]
        else:
            self.coords = torch.rand((N, 2), device=self.device) * 2 - 1
            # Temporal coordinates \in (0, 1), renormalized to the actual time
            # ranges of the initial conditions.
            t1, t2 = self.time_range
            int_times = torch.rand(N, device=self.device) * (t2 - t1) + t1  # Random times between t1 and t2

        X = torch.cat([
            torch.cat((self.coords, torch.full_like(int_times, 0).unsqueeze(1).to(self.device)), dim=1)
        ], dim=0) # (N, 3)
        X = torch.cat(
            (X, torch.hstack((self.coords, int_times.unsqueeze(1)))),
            dim=0
        ) # (2N, 3)
        
        return X

        # out_dict = {
        #     "coords": X,
        #     "src_kpts": self.src_kpts,
        #     "tgt_kpts": self.tgt_kpts
        # }
        # return out_dict
