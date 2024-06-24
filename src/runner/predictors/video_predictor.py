import torch
import logging
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
import os.path as osp


class VideoPredictor:
    def __init__(self, net, iframe_net, device, saved_dir, frame_dims, dataloader):
        self.device = device
        self.net = net.to(device)
        # self.res_net = res_net.to(device)
        self.iframe_net = iframe_net.to(self.device)
        self.saved_dir = saved_dir
        self.frame_dims = frame_dims
        self.dataloader = dataloader

    def predict(self):
        H, W = self.frame_dims
        self.net.eval()
        trange = tqdm(self.dataloader,
                        total=len(self.dataloader),
                        desc='Predicting')
        
        gt_frames = []
        pred_frames = []
        for batch in trange:
            batch = self._allocate_data(batch)
            coords, rgb, _, t = self._get_inputs_targets(batch)

            with torch.no_grad():
                # temporal warping
                coords = coords.squeeze() # (20000, 1)
                time_coord = torch.ones_like(coords[..., :1], dtype=torch.float32) * t  # (20000, 1)
                coords_time = torch.cat((coords, time_coord), dim=1).to(self.device)
                delta_coords = self.net(coords_time)["model_out"].squeeze() # (N, 2)
      
                # get the rgb values at time t
                outputs = self.iframe_net(coords + delta_coords, preserve_grad=True)["model_out"]

                # res_outputs = self.res_net(coords_time)["model_out"].squeeze()
                # outputs = outputs + res_outputs*0.1

                outputs = outputs.view(H, W, 3)
                # Clip the outputs to the valid range.
                outputs = torch.clamp(outputs, 0, 1)
                rgb = rgb.view(H, W, 3)
                gt_frames.append(rgb.detach().cpu().numpy())
                pred_frames.append(outputs.detach().cpu().numpy())

        # write the frames as video.mp4
        logging.info(f"Writing the video to {self.saved_dir}")
        self._write_video(gt_frames, osp.join(self.saved_dir, "gt.mp4"))
        self._write_video(pred_frames, osp.join(self.saved_dir, "pred.mp4"))

    def _get_inputs_targets(self, batch):
        """Specify the data input and target.
        Args:
            batch (dict): A batch of data.
        Returns:
            input (torch.Tensor): The data input.
            target (torch.LongTensor): The data target.
        """
        return batch['grid_coords'].detach().to(self.device).requires_grad_(False),\
              batch['rgb'].detach().to(self.device).requires_grad_(False),\
                  batch['idx'].detach().to(self.device).requires_grad_(False), \
                    batch['time_step']
    
    def _allocate_data(self, batch):
        """Allocate the data to the device.
        Args:
            batch (dict or sequence): A batch of the data.

        Returns:
            batch (dict or sequence): A batch of the allocated data.
        """
        if isinstance(batch, dict):
            return dict((key, self._allocate_data(data)) for key, data in batch.items())
        elif isinstance(batch, list):
            return list(self._allocate_data(data) for data in batch)
        elif isinstance(batch, tuple):
            return tuple(self._allocate_data(data) for data in batch)
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        
    def _write_video(self, frames, path):
        import cv2
        H, W = self.frame_dims
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, 30, (W, H))
        for frame in frames:
            frame = frame * 255
            frame = frame.astype('uint8')
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        cv2.destroyAllWindows()
        return path
