import torch

import click
import cv2
import glob
import os
import os.path as osp
from tqdm import tqdm
import yaml
import numpy as np

from data_preprocessing.data_preprocess import DataPreprocessor
from models import get_model
from resources.consts import IMAGE_EXTS
from utils.image_utils import tensor2img


def tensor_from_path(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))[None, :, :, :] / 255.
    img = (img * 2 - 1)
    img = torch.from_numpy(img).float()

    return img


@torch.no_grad()
@click.command()
@click.option('--source_root', type=str, required=True, help='Source root')
@click.option('--driver_root', type=str, required=True, help='Source root')
@click.option('--config_path', type=str, required=True, help='Config path')
@click.option('--model_path', type=str, required=True, help='Model path')
@click.option('--save_root', type=str, required=True, help='Save root')
@click.option('--skip_preprocess', is_flag=True, help='Do not use preprocessing')
def main(source_root, driver_root, config_path, model_path, save_root, skip_preprocess):
    '''
    Inference LP3D model. For each source image, render its novel views using a fixed camera trajectory
    '''

    # Preparing data
    device = 'cuda'
    processor = DataPreprocessor(device)

    if osp.isfile(source_root):
        source_paths = [source_root]
    else:
        source_paths = sorted(glob.glob(osp.join(source_root, '*')))
    source_paths = list(filter(lambda p: osp.splitext(p)[1][1:].lower() in IMAGE_EXTS, source_paths))

    if osp.isfile(driver_root):
        driver_paths = [driver_root]
    else:
        driver_paths = sorted(glob.glob(osp.join(driver_root, '*')))
    driver_paths = list(filter(lambda p: osp.splitext(p)[1][1:].lower() in IMAGE_EXTS, driver_paths))

    assert len(source_paths) > 0 and len(driver_paths) > 0, "No input image found"

    print('Preparing data...')
    all_source_data = []
    all_driver_data = []
    for source_path in tqdm(source_paths):
        if not skip_preprocess:
            source_data = processor.from_path(source_path, device, keep_bg=False)
            all_source_data.append(source_data)
        else:
            all_source_data.append({
                'image': tensor_from_path(source_path).to(device)
            })
    for driver_path in tqdm(driver_paths):
        if not skip_preprocess:
            driver_data = processor.from_path(driver_path, device, keep_bg=False)
            driver_data['exp_image'] = driver_data['image']
            all_driver_data.append(driver_data)
        else:
            all_driver_data.append({
                'exp_image': tensor_from_path(driver_path).to(device),
                'image': tensor_from_path(driver_path).to(device)
            })


    print(f'Number of pairs: {len(all_source_data)}')

    # Preparing model
    with open(config_path, 'r') as f:
        options = yaml.safe_load(f)
    model = get_model(options['model']).to(device)

    state_dict = torch.load(model_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    print(model_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Inference
    os.makedirs(save_root, exist_ok=True)

    for source_idx, source_data in enumerate(all_source_data):
        for driver_idx, driver_data in enumerate(all_driver_data):
            print(f'Processing {source_idx}/{driver_idx}')
            save_path = osp.join(save_root, f'{source_idx:04d}_{driver_idx}.png')

            out = model(
                xs_data=source_data,
                xd_data=driver_data,
            )

            out_hr = tensor2img(out['image'], min_max=(-1, 1))

            source_img = tensor2img(source_data['image'][0], min_max=(-1, 1))
            driver_img = tensor2img(driver_data['image'][0], min_max=(-1, 1))
            cv2.imwrite(save_path, np.hstack(
                (source_img, driver_img, out_hr)
            ))


if __name__ == '__main__':
    main()
