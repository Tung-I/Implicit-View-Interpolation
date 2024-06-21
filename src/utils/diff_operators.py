import torch
from torch.autograd import grad

def hessian(y, x):
    ''' hessian of y wrt x
    y: shape (num_observations, out_dim=2)
    x: shape (num_observations, in_dim=3)
    '''
    num_observations = y.shape[0]
    h = torch.zeros(num_observations, y.shape[-1], x.shape[-1], x.shape[-1]).to(y.device)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        dydx = gradient(y[..., i], x)  # (num_observations, in_dim)

        # calculate hessian on y for each x value
        for j in range(x.shape[-1]):
            h[..., i, j, :] = gradient(dydx[..., j], x)[..., :] # (num_observations, out_dim)
    return h


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)

def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def jacobian(y, x):
    ''' jacobian of y wrt x '''
    print(f"y.shape: {y.shape}")
    print(f"x.shape: {x.shape}")
    meta_batch_size, num_observations = y.shape[:2]
    jac = torch.zeros(meta_batch_size, num_observations, y.shape[-1], x.shape[-1]).to(y.device) # (meta_batch_size*num_points, 2, 2)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        y_flat = y[..., i].view(-1, 1)
        jac[:, :, i, :] = grad(y_flat, x, torch.ones_like(y_flat), create_graph=True)[0]

    status = 0
    if torch.any(torch.isnan(jac)):
        status = -1

    return jac, status
