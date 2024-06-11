import torch

import click
import cv2
import glob
import os
import os.path as osp
from tqdm import tqdm
import yaml
import numpy as np

from additional_modules.eg3d.camera_utils import IntrinsicsSampler, LookAtPoseSampler
from data_preprocessing.data_preprocess import DataPreprocessor
from models import get_model
from resources.consts import IMAGE_EXTS
from utils.image_utils import tensor2img


@torch.no_grad()
@click.command()
@click.option('--source_root', type=str, required=True, help='Source root')
@click.option('--config_path', type=str, required=True, help='Config path')
@click.option('--model_path', type=str, required=True, help='Model path')
@click.option('--save_root', type=str, required=True, help='Save root')
@click.option('--cam_batch_size', type=int, default=1, help='Batch size for cam2world')
@click.option('--skip_preprocess', is_flag=True, help='Do not use preprocessing')
def main(source_root, config_path, model_path, save_root, skip_preprocess, cam_batch_size):
    '''
    Inference LP3D model. For each source image, render its novel views using a fixed camera trajectory
    '''

    # Preparing data
    device = 'cuda'
    processor = DataPreprocessor(device)

    source_paths = sorted(glob.glob(osp.join(source_root, '*')))
    source_paths = list(filter(lambda p: osp.splitext(p)[1][1:].lower() in IMAGE_EXTS, source_paths))
    assert len(source_paths) > 0, "No input image found"

    # Preparing data
    print('Preparing data...')
    all_source_data = []
    for source_path in tqdm(source_paths):
        if not skip_preprocess:
            all_source_data.append(processor.from_path(source_path, device))
        else:
            img = cv2.imread(source_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = np.transpose(img, (2, 0, 1))[None, :, :, :] / 255.
            img = (img * 2 - 1)
            img = torch.from_numpy(img).float().to(device)
            all_source_data.append({
                'image': img
            })
    print(f'Number of sources: {len(all_source_data)}')

    # Preparing camera trajectory
    camera_lookat_point = torch.tensor([0, 0, 0.2]).float().to(device)
    yaw_range = 0.35
    pitch_range = 0.25
    num_keyframes = 50
    radius = 2.7

    trajectory_cam2worlds = []
    for view_idx in range(num_keyframes):
        yaw_angle = 3.14/2 + yaw_range * np.sin(2 * 3.14 * view_idx / num_keyframes)
        pitch_angle = 3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * view_idx / num_keyframes)
        trajectory_cam2worlds.append(
            LookAtPoseSampler.sample(
                yaw_angle, pitch_angle, radius,
                camera_lookat_point,
                yaw_angle, pitch_angle, 0,
                device=device
            )
        )
    
    intrinsics = IntrinsicsSampler.sample(
        18.837, 0.5,
        0, 0,
        batch_size=1,
        device=device
    )

    # Preparing model
    with open(config_path, 'r') as f:
        options = yaml.safe_load(f)
    model = get_model(options['model']).to(device)
    model.eval()

    state_dict = torch.load(model_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict, strict=False)

    # Inference
    os.makedirs(save_root, exist_ok=True)

    for source_idx, source_data in tqdm(enumerate(all_source_data), total=len(all_source_data)):
        frames = []

        for start_idx in range(0, len(trajectory_cam2worlds), cam_batch_size):
            batch_cam2world = trajectory_cam2worlds[start_idx: start_idx + cam_batch_size]
            all_xds_data = [{'cam2world': c, 'intrinsics': intrinsics} for c in batch_cam2world]
            out = model(
                xs_data=source_data,
                all_xds_data=all_xds_data
            )

            for x in out:
                frames.append(tensor2img(x['image'], min_max=(-1, 1)))

        save_path = osp.join(save_root, f'{source_idx:04d}.mp4')
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(save_path, fourcc, 30, (width, height))
        for frame in frames:
            video.write(frame)


if __name__ == '__main__':
    main()
