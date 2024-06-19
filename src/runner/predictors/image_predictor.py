import torch
import logging
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
import os.path as osp


class ImagePredictor:
    def __init__(self, net, metric_fns, device, saved_dir, im_dim, dataset):
        self.device = device
        self.net = net.to(device)
        self.metric_fns = [metric_fn.to(device) for metric_fn in metric_fns]
        self.saved_dir = saved_dir
        self.im_dim = im_dim
        self.dataset = dataset

    def predict(self):
        H, W = self.im_dim
        n_channels = 3
        self.net.eval()
        with torch.no_grad():
            # idx = torch.arange(H*W, device=self.device)
            # rec = torch.zeros((H, W, n_channels), device=self.device).requires_grad_(False)
            X = self.dataset.__getitem__()['grid_coords']
            X = X.detach().to(self.device).requires_grad_(False)
            rec = self.net(X, preserve_grad=True)["model_out"]
            rec = rec.detach().cpu().clip(0, 1).requires_grad_(False)

        sz = [H, W, n_channels]
        rec = rec.reshape(sz).permute((2, 0, 1))
        img = to_pil_image(rec)
        fname = self.dataset.path.split('/')[-1].split('.')[0]
        out_img_path = osp.join(self.saved_dir, fname + "_recon.png")
        img.save(out_img_path)
        print(f"Image saved as: {out_img_path}")

    def load(self, path):
        """Load the model checkpoint.
        Args:
            path (Path): The path to load the model checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint['net'])
