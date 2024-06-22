import torch
import torch.nn as nn
from src.utils.diff_operators import hessian

class DataConstraintLoss(nn.Module):
    """Data Constraint Loss for warping."""
    def __init__(self, intermediate_times=[0.25, 0.5, 0.75], constraint_weight=1e4):
        super(DataConstraintLoss, self).__init__()
        self.intermediate_times = intermediate_times
        self.constraint_weight = constraint_weight

    def forward(self, src_kpts, tgt_kpts, model):
        data_constraint = 0.0
        # Data fitting: f(src, 1)=tgt, f(tgt,-1)=src
        src = torch.cat((src_kpts, torch.ones_like(src_kpts[..., :1])), dim=1)
        y_src = model(src)['model_out']
        tgt = torch.cat((tgt_kpts, -torch.ones_like(tgt_kpts[..., :1])), dim=1)
        y_tgt = model(tgt)['model_out']
        data_constraint += (tgt_kpts - y_src)**2 + (src_kpts - y_tgt)**2
        data_constraint *= 1e2

        # Forcing the feature matching along time
        for t in self.intermediate_times:
            tgt_0 = torch.cat((tgt_kpts, (t-1)*torch.ones_like(tgt_kpts[..., :1])), dim=1)
            y_tgt_0 = model(tgt_0)['model_out']
            src_0 = torch.cat((src_kpts, t*torch.ones_like(src_kpts[..., :1])), dim=1)
            y_src_0 = model(src_0)['model_out']
            data_constraint += ((y_src_0 - y_tgt_0)**2)*5e1

            src_t = torch.cat((y_src_0, -t*torch.ones_like(src_kpts[..., :1])), dim=1)
            y_src_t = model(src_t)['model_out']
            data_constraint += ((y_src_t - src_kpts)**2)*2e1

            tgt_t = torch.cat((y_tgt_0, 1-t*torch.ones_like(tgt_kpts[..., :1])), dim=1)
            y_tgt_t = model(tgt_t)['model_out']
            data_constraint += ((y_tgt_t - tgt_kpts)**2)*2e1

        return data_constraint.mean()*self.constraint_weight

class IdentityConstraintLoss(nn.Module):
    """Identity Constraint Loss for warping."""
    def __init__(self, constraint_weight=1e3):
        super(IdentityConstraintLoss, self).__init__()
        self.constraint_weight = constraint_weight

    def forward(self, X, Y):
        # Identity constraint: f(p,0) = (p), penalize the diff if t=0
        diff_constraint = (Y - X[..., :2])**2  # (n_observation, 2)
        identity_constraint = torch.where(
            torch.cat((X[..., -1:], X[..., -1:]), dim=-1) == 0,
            diff_constraint,
            torch.zeros_like(diff_constraint)
        )

        return identity_constraint.mean()*self.constraint_weight

class InverseConstraintLoss(nn.Module):
    """Inverse Constraint Loss for warping."""
    def __init__(self, constraint_weight=1e4):
        super(InverseConstraintLoss, self).__init__()
        self.constraint_weight = constraint_weight
    
    def forward(self, X, Xs, Xt, X1):

        # Inverse constraint: f(f(p,t), 1-t) = f(p,1)
        # Ppenalize if t!=0
        inv_constraint = (Xs - X[..., 0:2])**2 + (Xt - X1)**2
        inv_constraint = torch.where(
            torch.cat((X[..., -1:], X[..., -1:]), dim=-1) == 0,
            torch.zeros_like(inv_constraint),
            inv_constraint
        )

        return inv_constraint.mean()*self.constraint_weight


class TPSConstraintLoss(nn.Module):
    """Thin Plate Spline Constraint Loss for warping."""
    def __init__(self, constraint_weight=1e3):
        super(TPSConstraintLoss, self).__init__()
        self.constraint_weight = constraint_weight

    def forward(self, X, Y):
        # M = model(coords)
        # X = M["model_in"]  # (n_observation, 3)
        # Y = M["model_out"].squeeze()  # (n_observation, 2)

        hessian1 = hessian(Y, X)  # (n_observation, y_dim, x_dim, x_dim)
        return (hessian1 ** 2).mean()*self.constraint_weight


