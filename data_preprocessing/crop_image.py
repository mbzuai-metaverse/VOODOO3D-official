import os.path as osp
from PIL import Image
import numpy as np

from additional_modules.deep3dfacerecon.util.load_mats import load_lm3d
from additional_modules.deep3dfacerecon.util.preprocess import align_img


TARGET_SIZE = 1024.
RESCALE_FACTOR = 300
CENTER_CROP_SIZE = 700
OUTPUT_SIZE = 512


class ImageCropper:
    def __init__(self, lm3d=None):
        bfm_folder = osp.join('additional_modules/deep3dfacerecon/BFM')
        if lm3d is None:
            self.lm3d_std = load_lm3d(bfm_folder)
        else:
            # calculate 5 facial landmarks using 68 landmarks
            lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
            lm3d_5p = np.stack([lm3d[lm_idx[0], :], np.mean(lm3d[lm_idx[[1, 2]], :], 0), np.mean(
                lm3d[lm_idx[[3, 4]], :], 0), lm3d[lm_idx[5], :], lm3d[lm_idx[6], :]], axis=0)
            lm3d_5p = lm3d_5p[[1, 2, 0, 3, 4], :]
            self.lm3d_std = lm3d_5p

    def __call__(self, im, lm):
        _, H = im.size
        lm = lm.copy()
        lm_orig = lm.copy()
        lm[:, -1] = H - 1 - lm[:, -1]

        _, im_high, _, lm_high, _, _ = align_img(
            im, lm, lm_orig, self.lm3d_std, target_size=TARGET_SIZE, rescale_factor=RESCALE_FACTOR
        )

        left = int(im_high.size[0]/2 - CENTER_CROP_SIZE/2)
        upper = int(im_high.size[1]/2 - CENTER_CROP_SIZE/2)
        right = left + CENTER_CROP_SIZE
        lower = upper + CENTER_CROP_SIZE
        im_cropped = im_high.crop((left, upper, right, lower))

        lm_cropped = lm_high.copy()
        lm_cropped[..., 0] -= left
        lm_cropped[..., 1] -= upper
        lm_cropped[..., 0] *= OUTPUT_SIZE / im_cropped.size[0]
        lm_cropped[..., 1] *= OUTPUT_SIZE / im_cropped.size[1]

        im_cropped = im_cropped.resize((OUTPUT_SIZE, OUTPUT_SIZE), resample=Image.LANCZOS)

        return im_cropped, lm_cropped
