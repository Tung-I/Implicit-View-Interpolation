import os
import sys
import shutil
import json
import gc
from pathlib import Path
from timeit import default_timer as timer
import math
import warnings
warnings.filterwarnings('ignore')
import torch
from PIL import Image
import numpy as np

sys.path.append('/home/tvchen/SimpleSfm')
from simple_sfm.scene_utils.video_to_scene_processors import OneVideoSceneProcesser
from simple_sfm.scene_utils.matcher import Matcher
from simple_sfm.scene_utils.colmap_scene_converters import colmap_sparse_to_re10k_like_views
from simple_sfm.scene_utils.colmap_bd_utils import ColmapBdManager
from simple_sfm.scene_utils.scene_readers import read_re10k_views
from simple_sfm.utils.video_streamer import VideoStreamer
from simple_sfm.cameras.camera_multiple import CameraMultiple
from simple_sfm.cameras.utils import average_extrinsics

sys.path.append('/home/tvchen/MLI')
from lib.trainers.utils import create_trainer_load_weights_from_config
from lib.utils.io import get_config

video1_path = Path("/dlbimg/datasets/View_transition/content_banjoman_960x540/1s_video/1s_vw_000.mp4")
video2_path = Path("/dlbimg/datasets/View_transition/content_banjoman_960x540/1s_video/1s_vw_002.mp4")
plot_size = 1


def save_mpi(mpi, path, save_as_jpg=False):
        n_planes, channels, height, width = mpi.shape
        mpi = np.concatenate([mpi[:, :-1] * 0.5 + 0.5, mpi[:, -1:]], axis=1)
        mpi = mpi.transpose((0, 2, 3, 1))
        mpi = (255 * np.clip(mpi, 0, 1)).astype(np.uint8)

        os.makedirs(path, exist_ok=True)
        for i, layer in enumerate(mpi):
            if save_as_jpg:
                Image.fromarray(layer[:, :, :3]).save(os.path.join(path, f'layer_{i:02d}.jpg'), optimize=True)
                Image.fromarray(np.repeat(layer[:, :, -1:], 3, axis=2)).save(os.path.join(path, f'layer_alpha_{i:02d}.jpg'), optimize=True)
            else:
                Image.fromarray(layer).save(os.path.join(path, f'layer_{i:02d}.png'), optimize=True)

def save_layered_depth(layered_depth, path):
  num_layers, h, w = layered_depth.shape
  depth_meta_data = {}
  for i in range(num_layers):
      depth = layered_depth[i]
      low, high = np.min(depth), np.max(depth)
      scaled = (depth - low) / (high - low)
      ui = np.clip(scaled * 256.0, 0, 255).astype(np.uint8)
      img = Image.fromarray(ui)

      tag = f'layer_depth_{i:02d}'
      img.save(os.path.join(path, tag + '.jpg'), quality=100)
      depth_meta_data[tag] = [low.item(), high.item()]

  return depth_meta_data

def write_meta(depth_meta_data, resolution, extrinsic, intrinsic, path):
      meta_data = depth_meta_data

      meta_data.update({'frame_size': resolution,
                        'extrinsic_re':extrinsic,
                        'intrinsics_re':intrinsic,
                        })

      with open(os.path.join(path,'meta.json'), 'w') as f:
        json.dump(meta_data, f, ensure_ascii=True)

def find_most_distant_point_a_to_b(set_a, set_b):
    size_a = set_a.shape[0]
    size_b = set_b.shape[0]
    distances = torch.sqrt(torch.sum((set_b.repeat(size_a, 1) - set_a.repeat(1, size_b).reshape(-1, 3))**2, dim=1))
    min_dist, _ = distances.reshape(size_a, -1).min(dim=1)
    return torch.argmax(min_dist)

def find_most_distant_point(point_set):
    size_point_set = point_set.shape[0]
    distances = torch.sqrt(torch.sum((point_set.repeat(size_point_set, 1) - point_set.repeat(1, size_point_set).reshape(-1, 3))**2, dim=1))
    mask = (distances == 0)
    distances = distances + mask * max(distances)
    min_dist, _ = distances.reshape(size_point_set, -1).min(dim=1)
    return torch.argmax(min_dist)

def get_k_most_distant_cams(cameras, k):
    cam_pos = cameras.world_position
    cam_dir = cameras.world_view_direction()

    first_point_id = find_most_distant_point(cam_pos)
    result = cam_pos[[first_point_id]][None]
    result_id = first_point_id[None]

    k = num_views
    for i in range(k - 1):
        next_point_id = find_most_distant_point_a_to_b(cam_pos, cam_pos[result_id])
        result_id = torch.cat([result_id, next_point_id[None]])

    return cameras[result_id], result_id


if __name__ == "__main__":
    print('Loading pretrained model...')
    num_layers = '8' #@param ['4', '8']

    checkpoints_path = f'pretrained/model{num_layers}_layers/'
    config_path = f'pretrained/model{num_layers}_layers/tblock{num_layers}.yaml'

    config = get_config(config_path)
    iteration = 660000

    trainer, loaded_iteration = create_trainer_load_weights_from_config(config=config,
                                                                        checkpoints_dir=checkpoints_path,
                                                                        iteration=iteration,
                                                                        device='cuda'
                                                                        )

    _ = trainer.eval()

    # print('Loading videos...')

    # skip = 1
    # max_num_frames = None  # @param {type: 'integer'}
    # center_crop = False  # @param {type: 'boolean'}
    # scale_factor = 1  #@param {type:"slider", min:0.1, max:1, step:0.05}

    # video_path = video2_path
    # capture_work_dir = Path("output/view2")

    # video_to_frames = OneVideoSceneProcesser(
    #     video_path=str(video_path),
    #     dataset_output_path=str(capture_work_dir),
    #     skip=skip,
    #     center_crop=center_crop,
    #     scale_factor=scale_factor,
    #     max_len=max_num_frames,
    #     img_prefix='jpg',
    #     filter_with_sharpness=True,
    # )

    # video_to_frames.run()
    # frames_path = Path(capture_work_dir, 'frames')

    # print("Run COLMAP feature matching...")

    capture_work_dir = Path("output/v1v2")
    frames_path = Path(capture_work_dir, 'frames')
    superglue_weigths_path = '/home/tvchen/SuperGluePretrainedNetwork/models/weights/superglue_outdoor.pth'
    superpoint_weigths_path = '/home/tvchen/SuperGluePretrainedNetwork/models/weights/superpoint_v1.pth'

    # @markdown Non Maximum Suppression (NMS) radius
    nms_radius = 4  #@param {type:"slider", min:1, max:10, step:1}
    # @markdown Detector confidence threshold.
    keypoint_threshold = 0.01  #@param {type:"slider", min:0.001, max:0.1, step:0.001}
    # @markdown Threshold value for matching.
    match_threshold = 0.75  #@param {type:"slider", min:0.05, max:0.95, step:0.05}
    # @markdown Num sinkhorn iterations for super glue matching.
    sinkhorn_iterations = 20 #@param {type:"slider", min:1, max:50, step:1}
    # @markdown Batch size for super glue infer
    super_glue_batch = 5 #@param {type:"slider", min:1, max:10, step:1}

    matcher = Matcher(
        super_point_extractor_weights_path=superpoint_weigths_path,
        super_glue_weights_path=superglue_weigths_path,
        nms_radius=nms_radius,
        keypoint_threshold=keypoint_threshold,
        matcher_type='super_glue',
        match_threshold=match_threshold,
        sinkhorn_iterations=sinkhorn_iterations,
        super_glue_batch=int(super_glue_batch)
        )
    
    vs = VideoStreamer(
        str(frames_path),
        height=None,
        width=None,
        max_len=None,
        img_glob='*.jpg'
    )
    camera_size = vs.get_resolution()

    colmap = ColmapBdManager(
        db_dir=str(Path(capture_work_dir, 'colmap')),
        images_folder_path=str(frames_path),
        camera_type='OPENCV',
        camera_params=None,
        camera_size=camera_size
    )

    processed_frames, match_table, images_names = matcher.match_video_stream(vs, lambda x: True)
    torch.cuda.empty_cache()

    colmap.replace_images_data(images_names)
    colmap.replace_keypoints(images_names, processed_frames)
    colmap.replace_and_verificate_matches(match_table, images_names)
    num_sparse_points = colmap.run_mapper()

    print("Convert COLMAP sparse to re10k like views...")

    colmap_sparse_to_re10k_like_views(
        scene_colmap_sparse_path=os.path.join(capture_work_dir, 'colmap', 'sparse'),
        views_file_output_path=capture_work_dir,
        scene_meta_file_output_path=capture_work_dir,
    )

    # # @markdown Source camera resolution rescale:
    # source_camera_scale = 1  #@param {type:"slider", min:0.1, max:1, step:0.05}

    source_camera_scale = 1

    # @markdown Scene scale:
    translation_scale = 2 #@param {type:"slider", min:1, max:5, step:0.1}

    resize_size = [int(camera_size[1] * source_camera_scale),
                int(camera_size[0] * source_camera_scale)]


    # resize_size = [math.floor(resize_size[0] / 16) * 16,
    #                math.floor(resize_size[1] / 16) * 16]


    crop_size = [math.floor(resize_size[0] / 16) * 16,
                math.floor(resize_size[1] / 16) * 16]

    intrinsics, extrinsics, images = read_re10k_views(
        views_file_path=os.path.join(capture_work_dir, 'views.txt'),
        scene_meta_path=os.path.join(capture_work_dir, 'scene_meta.yaml'),
        frames_path=os.path.join(capture_work_dir, 'frames'),
        frames_crop_size=crop_size,
        frames_resize_size=resize_size,
        translation_scale=translation_scale
        )



    extrinsics = torch.stack(extrinsics, dim=0).cuda()
    intrinsics = torch.stack(intrinsics, dim=0).cuda()
    images = torch.stack(images, dim=0).cuda()
    source_images_sizes = images.shape[-2:]

    all_cameras = CameraMultiple(extrinsics=extrinsics,
                                intrinsics=intrinsics,
                                images_sizes=torch.stack([torch.tensor(images.shape[-2:])] * extrinsics.shape[0], dim=0).cuda()
                                )


    # @markdown Select number of views for bulding MLI
    num_views = 2  #@param {type:"slider", min:1, max:10, step:1}

    

    selected_cameras, select_id = get_k_most_distant_cams(all_cameras, num_views)
    selected_images = images[select_id]

    # selected_cameras = all_cameras
    # selected_images = images

    reference_extrinsic = average_extrinsics(selected_cameras.extrinsics)

    # # @markdown Reference camera scale
    # ref_camera_scale = 2  #@param {type:"slider", min:0.5, max:2, step:0.05}

    ref_camera_scale = 1
    ref_intrinsic = intrinsics[:1].clone()
    reference_images_sizes = [int(source_images_sizes[0] * ref_camera_scale),
                            int(source_images_sizes[1] * ref_camera_scale)]
    reference_images_sizes = [reference_images_sizes[0] // 16 * 16, reference_images_sizes[1] // 16 * 16]
    ref_intrinsic[:, :2, :2] = ref_intrinsic[:, :2, :2] / ref_camera_scale


    # reference_images_sizes = [720, 1280]
    # reference_images_sizes = [reference_images_sizes[0] // 16 * 16, reference_images_sizes[1] // 16 * 16]
    # ref_intrinsic = torch.tensor([[[reference_images_sizes[0] / reference_images_sizes[1], 0, 0.5],
    #                                [0, 1.0, 0.5],
    #                                [0, 0, 1.0]]]).cuda()

    reference_camera = CameraMultiple(extrinsics=reference_extrinsic[None, None],
                                    intrinsics=ref_intrinsic[None, None],
                                    images_sizes=reference_images_sizes,
                                    )
    

    print('Run MLI...')
    torch.cuda.empty_cache()

    trainer.eval()

    with torch.no_grad():
        result = trainer.gen.manual_forward(source_images=selected_images[None],
                                            source_cameras=selected_cameras[None, None],
                                            reference_cameras=reference_camera
                                            )
    

    geom_path = 'output/multi-layer-viewer/mli_scene/'

    mpi = result['mpi'][0, 0].cpu()
    layered_depth = result['layered_depth'][0].cpu()

    save_mpi(mpi, geom_path, save_as_jpg=True)
    depth_meta_data = save_layered_depth(layered_depth.numpy(), geom_path)

    intr = reference_camera.intrinsics[0].flatten().cpu().numpy()
    extr = reference_camera.extrinsics[0].flatten().cpu().numpy()
    write_meta(depth_meta_data,
            [layered_depth.shape[1], layered_depth.shape[2]],
            list(extr.astype(float)),
            list(intr.astype(float)),
            geom_path)

    final_mli_path = os.path.join(capture_work_dir, f'_mli.zip')
    final_geom_path = os.path.join(geom_path, '*')

    print(f'You can find saved MLI here :{final_mli_path}')
