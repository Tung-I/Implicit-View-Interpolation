import torch
import torch.nn as nn
from src.utils.diff_operators import hessian

DEFAULT_WEIGHTS_FMLOSS = {
    "data_constraint":  1e4,
    "identity_constraint":  1e3,
    "inv_constraint": 1e4,
    "TPS_constraint": 1e3,
}


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


# class WarpingLoss(torch.nn.Module):
#     """Warping loss with feature matching between source and target.

#     Parameters
#     ----------
#     warp_src_pts: torch.Tensor
#         An Nx2 tensor with the feature locations in the source image. Note that
#         these points must be normalized to the [-1, 1] range.

#     warp_tgt_pts: torch.Tensor
#         An Nx2 tensor with the feature locations in the target image. Note that
#         these points must be normalized to the [-1, 1] range.

#     intermediate_times: list, optional
#         List of intermediate times where the data constraint will be fit. All
#         values must be in range [0, 1]. By default is [0.25, 0.5, 0.75]

#     constraint_weights: dict, optional
#         The weight of each constraint in the final loss composition. By
#         default, the weights are:
#         {
#             "data_constraint":  1e4,
#             "identity_constraint":  1e3,
#             "inv_constraint": 1e4,
#             "TPS_constraint": 1e3,
#         }
#     """
#     def __init__(
#             self,
#             intermediate_times: list = [0.25, 0.5, 0.75],
#             constraint_weights: dict = DEFAULT_WEIGHTS_FMLOSS
#     ):
#         super(WarpingLoss, self).__init__()
#         self.intermediate_times = intermediate_times
#         self.constraint_weights = constraint_weights
#         if intermediate_times is None or not len(intermediate_times):
#             self.intermediate_times = [0.25, 0.5, 0.75]
#         if constraint_weights is None or not len(constraint_weights):
#             self.constraint_weights = DEFAULT_WEIGHTS_FMLOSS

#         # Ensuring that all necessary weights are stored.
#         for k, v in DEFAULT_WEIGHTS_FMLOSS.items():
#             if k not in self.constraint_weights:
#                 self.constraint_weights[k] = v

#     def forward(self, coords, src_kpts, tgt_kpts, model):
#         """
#         coords: torch.Tensor(shape=[N, 3])
#         model: torch.nn.Module
#         """
#         # print(coords.requires_grad)
#         M = model(coords)
#         X = M["model_in"] #(n_observation, 3)
#         Y = M["model_out"].squeeze() #(n_observation, 2)

#         # #Print X/Y requires grad
#         # print(X.requires_grad)
#         # print(Y.requires_grad)

#         # Thin plate spline energy
#         hessian1 = hessian(Y, X) #(n_observation, y_dim, x_dim, x_dim)
#         TPS_constraint = hessian1 ** 2

#         # Data constraint
#         # Data fitting: f(src, 1)=tgt, f(tgt,-1)=src
#         src = torch.cat((src_kpts, torch.ones_like(src_kpts[..., :1])), dim=1)
#         y_src = model(src)['model_out']
#         tgt = torch.cat((tgt_kpts, -torch.ones_like(tgt_kpts[..., :1])), dim=1)
#         y_tgt = model(tgt)['model_out']
#         data_constraint = (tgt_kpts - y_src)**2 + (src_kpts - y_tgt)**2
#         data_constraint *= 1e2

#         # Forcing the feature matching along time
#         for t in self.intermediate_times:
#             tgt_0 = torch.cat((tgt_kpts, (t-1)*torch.ones_like(tgt_kpts[..., :1])), dim=1)
#             y_tgt_0 = model(tgt_0)['model_out']
#             src_0 = torch.cat((src_kpts, t*torch.ones_like(src_kpts[..., :1])), dim=1)
#             y_src_0 = model(src_0)['model_out']
#             data_constraint += ((y_src_0 - y_tgt_0)**2)*5e1

#             src_t = torch.cat((y_src_0, -t*torch.ones_like(src_kpts[..., :1])), dim=1)
#             y_src_t = model(src_t)['model_out']
#             data_constraint += ((y_src_t - src_kpts)**2)*2e1

#             tgt_t = torch.cat((y_tgt_0, 1-t*torch.ones_like(tgt_kpts[..., :1])), dim=1)
#             y_tgt_t = model(tgt_t)['model_out']
#             data_constraint += ((y_tgt_t - tgt_kpts)**2)*2e1

#         # Identity constraint: f(p,0) = (p), penalize the diff if t=0
#         diff_constraint = (Y - X[..., :2])**2 #(n_observation, 2)
#         identity_constraint = torch.where(torch.cat((coords[..., -1:], coords[..., -1:]), dim=-1) == 0, diff_constraint, torch.zeros_like(diff_constraint))

#         # Inverse constraint: f(f(p,t), -t) = p,  f(f(p,-t), t) = p
#         Ys = torch.cat((Y, -X[..., -1:]), dim=1)
#         model_Xs = model(Ys)
#         Xs = model_Xs['model_out']

#         # Inverse constraint: f(f(p,t), 1-t) = f(p,1)
#         Yt = torch.cat((Y, 1 - X[..., -1:]), dim=1)
#         model_Xt = model(Yt)
#         Xt = model_Xt['model_out']
#         Y1 = torch.cat((X[...,0:2], torch.ones_like(X[..., -1:])), dim=1)
#         X1 = model(Y1)['model_out']

#         # Inverse constraint: penalize if t!=0
#         inv_constraint = (Xs - X[..., 0:2])**2 + (Xt - X1)**2
#         inv_constraint = torch.where(torch.cat((coords[..., -1:], coords[..., -1:]), dim=-1) == 0, torch.zeros_like(inv_constraint), inv_constraint)

#         # return {
#         #     "data_constraint": data_constraint.mean() * self.constraint_weights["data_constraint"],
#         #     "identity_constraint": identity_constraint.mean() * self.constraint_weights["identity_constraint"],
#         #     "inv_constraint": inv_constraint.mean() * self.constraint_weights["inv_constraint"],
#         #     "TPS_constraint": TPS_constraint.mean() * self.constraint_weights["TPS_constraint"],
#         # }
    
#         return data_constraint.mean() * self.constraint_weights["data_constraint"] + \
#             identity_constraint.mean() * self.constraint_weights["identity_constraint"] + \
#                 inv_constraint.mean() * self.constraint_weights["inv_constraint"] + \
#                     TPS_constraint.mean() * self.constraint_weights["TPS_constraint"]