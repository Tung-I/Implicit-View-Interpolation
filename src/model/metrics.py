import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter


class PSNR(nn.Module):
    """The Peak Signal-to-Noise Ratio (PSNR) metric.
    """
    def __init__(self, max_val=1.0):
        super().__init__()
        self.max_val = max_val

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor) (N, C, H, W): The model output.
            target (torch.Tensor) (N, C, H, W): The ground truth target.
        Returns:
            metric (torch.Tensor) (0): The PSNR score.
        """
        mse = F.mse_loss(output, target, reduction='none').mean(dim=(1, 2, 3))
        psnr = 20 * torch.log10(self.max_val / torch.sqrt(mse))
        return psnr.mean()

def ssim(img1, img2, max_val):
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    mu1 = gaussian_filter(img1, 1.5)
    mu2 = gaussian_filter(img2, 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter(img1 ** 2, 1.5) - mu1_sq
    sigma2_sq = gaussian_filter(img2 ** 2, 1.5) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

class SSIM(nn.Module):
    """The Structural Similarity (SSIM) metric.
    """
    def __init__(self, max_val=1.0):
        super().__init__()
        self.max_val = max_val

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor) (N, C, H, W): The model output.
            target (torch.Tensor) (N, C, H, W): The ground truth target.
        Returns:
            metric (torch.Tensor) (0): The SSIM score.
        """
        ssim_scores = []
        for i in range(output.shape[0]):
            output_img = output[i].cpu().numpy().transpose(1, 2, 0)
            target_img = target[i].cpu().numpy().transpose(1, 2, 0)
            ssim_score = ssim(output_img, target_img, self.max_val)
            ssim_scores.append(ssim_score)
        return torch.tensor(ssim_scores).mean()