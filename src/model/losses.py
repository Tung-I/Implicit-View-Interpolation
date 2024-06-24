import torch
import torch.nn as nn
from src.utils.diff_operators import hessian
import torchvision.models as models
import pytorch_ssim
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

DEFAULT_WEIGHTS_FMLOSS = {
    "data_constraint":  1e4,
    "identity_constraint":  1e3,
    "inv_constraint": 1e4,
    "TPS_constraint": 1e3,
}

def get_outnorm(x:torch.Tensor, out_norm:str='') -> torch.Tensor:
    """ Common function to get a loss normalization value. Can
        normalize by either the batch size ('b'), the number of
        channels ('c'), the image size ('i') or combinations
        ('bi', 'bci', etc)
    """
    # b, c, h, w = x.size()
    img_shape = x.shape

    if not out_norm:
        return 1

    norm = 1
    if 'b' in out_norm: # normalize by batch size
        norm /= img_shape[0]
    if 'c' in out_norm: # normalize by the number of channels
        norm /= img_shape[-3]
    if 'i' in out_norm: # normalize by image/map size
        norm /= img_shape[-1]*img_shape[-2]

    return norm

class ColorConsistencyLoss(nn.Module):
    """The Color Consistency Loss.
    """
    def __init__(self):
        super(ColorConsistencyLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, output, target):
        output_lab = self.rgb_to_lab(output)
        target_lab = self.rgb_to_lab(target)
        return self.mse_loss(output_lab, target_lab)

    @staticmethod
    def rgb_to_lab(img):
        # Convert RGB image to Lab color space
        img = img.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        img = img.cpu().detach().numpy()
        lab_imgs = []
        for i in range(img.shape[0]):
            lab_img = cv2.cvtColor(img[i], cv2.COLOR_RGB2Lab)
            lab_imgs.append(lab_img)
        lab_imgs = torch.tensor(lab_imgs).permute(0, 3, 1, 2).float()  # (N, H, W, C) -> (N, C, H, W)
        return lab_imgs.to(img.device)

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-6, out_norm:str='bci'):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.out_norm = out_norm

    def forward(self, x, y):
        norm = get_outnorm(x, self.out_norm)
        loss = torch.sum(torch.sqrt((x - y).pow(2) + self.eps**2))
        return loss*norm

class PerceptualLoss(nn.Module):
    def __init__(self, feature_layers=[0, 5, 10, 19, 28]):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.features = nn.ModuleList([vgg[i] for i in feature_layers]).eval()
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, output, target):
        loss = 0.0
        for feature in self.features:
            output = feature(output)
            target = feature(target)
            loss += nn.functional.mse_loss(output, target)
        return loss

class StyleLoss(nn.Module):
    def __init__(self, feature_layers=[0, 5, 10, 19, 28]):
        super(StyleLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.features = nn.ModuleList([vgg[i] for i in feature_layers]).eval()
        for param in self.features.parameters():
            param.requires_grad = False

    def gram_matrix(self, input):
        b, c, h, w = input.size()
        features = input.view(b, c, h * w)
        G = torch.bmm(features, features.transpose(1, 2))
        return G.div(c * h * w)

    def forward(self, output, target):
        loss = 0.0
        for feature in self.features:
            output = feature(output)
            target = feature(target)
            loss += nn.functional.mse_loss(self.gram_matrix(output), self.gram_matrix(target))
        return loss

class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
        self.ssim_module = SSIM(data_range=1.0, size_average=True, channel=3)

    def forward(self, output, target):
        return 1 - self.ssim_module(output, target)

class WarpingLoss(torch.nn.Module):
    """Warping loss with feature matching between source and target.

    Parameters
    ----------
    warp_src_pts: torch.Tensor
        An Nx2 tensor with the feature locations in the source image. Note that
        these points must be normalized to the [-1, 1] range.

    warp_tgt_pts: torch.Tensor
        An Nx2 tensor with the feature locations in the target image. Note that
        these points must be normalized to the [-1, 1] range.

    intermediate_times: list, optional
        List of intermediate times where the data constraint will be fit. All
        values must be in range [0, 1]. By default is [0.25, 0.5, 0.75]

    constraint_weights: dict, optional
        The weight of each constraint in the final loss composition. By
        default, the weights are:
        {
            "data_constraint":  1e4,
            "identity_constraint":  1e3,
            "inv_constraint": 1e4,
            "TPS_constraint": 1e3,
        }
    """
    def __init__(
            self,
            warp_src_pts: torch.Tensor,
            warp_tgt_pts: torch.Tensor,
            intermediate_times: list = [0.25, 0.5, 0.75],
            constraint_weights: dict = DEFAULT_WEIGHTS_FMLOSS
    ):
        super(WarpingLoss, self).__init__()
        self.src = warp_src_pts
        self.tgt = warp_tgt_pts
        self.intermediate_times = intermediate_times
        self.constraint_weights = constraint_weights
        if intermediate_times is None or not len(intermediate_times):
            self.intermediate_times = [0.25, 0.5, 0.75]
        if constraint_weights is None or not len(constraint_weights):
            self.constraint_weights = DEFAULT_WEIGHTS_FMLOSS

        # Ensuring that all necessary weights are stored.
        for k, v in DEFAULT_WEIGHTS_FMLOSS.items():
            if k not in self.constraint_weights:
                self.constraint_weights[k] = v

    def forward(self, coords, model):
        """
        coords: torch.Tensor(shape=[N, 3])
        model: torch.nn.Module
        """
        # print(coords.requires_grad)
        M = model(coords)
        X = M["model_in"] #(n_observation, 3)
        Y = M["model_out"].squeeze() #(n_observation, 2)

        # #Print X/Y requires grad
        # print(X.requires_grad)
        # print(Y.requires_grad)

        # Thin plate spline energy
        hessian1 = hessian(Y, X) #(n_observation, y_dim, x_dim, x_dim)
        TPS_constraint = hessian1 ** 2

        # Data constraint
        # Data fitting: f(src, 1)=tgt, f(tgt,-1)=src
        src = torch.cat((self.src, torch.ones_like(self.src[..., :1])), dim=1)
        y_src = model(src)['model_out']
        tgt = torch.cat((self.tgt, -torch.ones_like(self.tgt[..., :1])), dim=1)
        y_tgt = model(tgt)['model_out']
        data_constraint = (self.tgt - y_src)**2 + (self.src - y_tgt)**2
        data_constraint *= 1e2

        # Forcing the feature matching along time
        for t in self.intermediate_times:
            tgt_0 = torch.cat((self.tgt, (t-1)*torch.ones_like(self.tgt[..., :1])), dim=1)
            y_tgt_0 = model(tgt_0)['model_out']
            src_0 = torch.cat((self.src, t*torch.ones_like(self.src[..., :1])), dim=1)
            y_src_0 = model(src_0)['model_out']
            data_constraint += ((y_src_0 - y_tgt_0)**2)*5e1

            src_t = torch.cat((y_src_0, -t*torch.ones_like(self.src[..., :1])), dim=1)
            y_src_t = model(src_t)['model_out']
            data_constraint += ((y_src_t - self.src)**2)*2e1

            tgt_t = torch.cat((y_tgt_0, 1-t*torch.ones_like(self.tgt[..., :1])), dim=1)
            y_tgt_t = model(tgt_t)['model_out']
            data_constraint += ((y_tgt_t - self.tgt)**2)*2e1

        # Identity constraint: f(p,0) = (p), penalize the diff if t=0
        diff_constraint = (Y - X[..., :2])**2 #(n_observation, 2)
        identity_constraint = torch.where(torch.cat((coords[..., -1:], coords[..., -1:]), dim=-1) == 0, diff_constraint, torch.zeros_like(diff_constraint))

        # Inverse constraint: f(f(p,t), -t) = p,  f(f(p,-t), t) = p
        Ys = torch.cat((Y, -X[..., -1:]), dim=1)
        model_Xs = model(Ys)
        Xs = model_Xs['model_out']

        # Inverse constraint: f(f(p,t), 1-t) = f(p,1)
        Yt = torch.cat((Y, 1 - X[..., -1:]), dim=1)
        model_Xt = model(Yt)
        Xt = model_Xt['model_out']
        Y1 = torch.cat((X[...,0:2], torch.ones_like(X[..., -1:])), dim=1)
        X1 = model(Y1)['model_out']

        # Inverse constraint: penalize if t!=0
        inv_constraint = (Xs - X[..., 0:2])**2 + (Xt - X1)**2
        inv_constraint = torch.where(torch.cat((coords[..., -1:], coords[..., -1:]), dim=-1) == 0, torch.zeros_like(inv_constraint), inv_constraint)

        return {
            "data_constraint": data_constraint.mean() * self.constraint_weights["data_constraint"],
            "identity_constraint": identity_constraint.mean() * self.constraint_weights["identity_constraint"],
            "inv_constraint": inv_constraint.mean() * self.constraint_weights["inv_constraint"],
            "TPS_constraint": TPS_constraint.mean() * self.constraint_weights["TPS_constraint"],
        }

