import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from typing import Dict

from additional_modules.deeplabv3.deeplabv3 import DeepLabV3
from additional_modules.eg3d.camera_utils import IntrinsicsSampler, LookAtPoseSampler
from additional_modules.segformer.backbone import Block, OverlapPatchEmbed
from models.utils.face_augmentor import FaceAugmentor
from models.lp3d_model import PositionalEncoder, Lp3D
from utils.registry import MODEL_REGISTRY


class ExpEncoder(PositionalEncoder):
    def __init__(self, img_size=512, img_channels=3, use_aug=True):
        super().__init__(img_size)

        self.use_aug = use_aug

        self.source_feat_extractor = DeepLabV3(input_channels=img_channels + 2)
        self.driver_feat_extractor = DeepLabV3(input_channels=img_channels + 2)
        self.triplane_descriptor = nn.Sequential(
            nn.Conv2d(96, 96, 3, 2, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(96, 96, 3, 1, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(96, 128, 3, 2, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1, bias=True),
        )

        self.patch_embed = OverlapPatchEmbed(
            img_size=img_size // 8, patch_size=3, stride=2, in_chans=256 * 2 + 128, embed_dim=1024
        )
        self.block1 = Block(dim=1024, num_heads=4, mlp_ratio=2, sr_ratio=1)
        self.block2 = Block(dim=1024, num_heads=4, mlp_ratio=2, sr_ratio=1)

        self.up1 = nn.PixelShuffle(upscale_factor=2)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv1 = nn.Conv2d(256, 128, 3, 1, 1, bias=True)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=True)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(128, 96, 3, 1, 1, bias=True)

        self.driver_aug = FaceAugmentor()

    def _calculate_delta(self, xs_img, xd_img, xs_triplane):
        if self.use_aug:
            xd_img = self.driver_aug(
                xd_img,
                target_size=(512, 512),
                apply_color_aug=self.training,
                apply_rnd_mask=self.training,
                apply_rnd_zoom=self.training,
            )
        else:
            xd_img = F.interpolate(xd_img, xs_img.shape[-2:])

        xs_img = self._add_positional_encoding(xs_img)
        xd_img = self._add_positional_encoding(xd_img)

        xs_feat = self.source_feat_extractor(xs_img)
        xd_feat = self.driver_feat_extractor(xd_img)

        xs_triplane = xs_triplane.reshape(-1, 96, 256, 256)
        xs_triplane_feat = self.triplane_descriptor(xs_triplane)
        x = torch.cat((xs_feat, xd_feat, xs_triplane_feat),  dim=1)

        x, H, W = self.patch_embed(x)
        x = self.block1(x, H, W)
        x = self.block2(x, H, W)
        x = x.reshape(xs_img.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = self.up1(x)
        x = self.up2(x)

        x = self.conv1(x)
        x = self.act1(x)
        x = self.up3(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)

        x = x.reshape(-1, 3, 32, 256, 256)

        return x

    def forward(self, xs_img, xd_img, xs_triplane):
        delta = self._calculate_delta(xs_img, xd_img, xs_triplane)

        xs_triplane = xs_triplane + delta

        return xs_triplane


@MODEL_REGISTRY.register()
class Voodoo3D(Lp3D):
    def __init__(
        self,
        neural_rendering_resolution: int,  # Render at this resolution and use superres to upsample to 512x512
        triplane_nd: int,  # Triplane's number of channels
        triplane_h: int,  # Triplane height
        triplane_w: int,  # Triplane width
        use_aug: bool,
        rendering_kwargs,
        superresolution_kwargs,
    ):
        self.use_aug = use_aug

        super().__init__(
            neural_rendering_resolution,
            triplane_nd,
            triplane_h,
            triplane_w,
            rendering_kwargs,
            superresolution_kwargs,
        )

        lookat_point = torch.tensor(rendering_kwargs['lookat_point']).unsqueeze(0).float()
        canonical_cam2world = LookAtPoseSampler.sample(
            np.pi / 2, np.pi / 2, rendering_kwargs['camera_radius'],
            lookat_point,
            np.pi / 2, np.pi / 2, 0.0,
            batch_size=1
        )
        canonical_intrinsics = IntrinsicsSampler.sample(
            18.837, 0.5,
            0, 0,
            batch_size=1
        )

        self.register_buffer('lookat_point', lookat_point)
        self.register_buffer('canonical_cam2world', canonical_cam2world)
        self.register_buffer('canonical_intrinsics', canonical_intrinsics)

    def _setup_modules(self):
        super()._setup_modules()
        self.exp_transfer = ExpEncoder(use_aug=self.use_aug)

        self.triplane_encoder.requires_grad_(False)
        self.superresolution.requires_grad_(False)
        self.decoder.requires_grad_(False)

    def frontalize(self, inp, neural_upsample):
        """
            Frontalize the input image/triplane.

            Parameters:
                - inp (Tensor): Can be either image (Bx3xHxW) or triplane (Bx3x32x256x256)
                - neural_upsample (bool): If True, use superresolution to upsample the output. Otherwise, just
                upsample it using nearest interpolation.
        """

        if len(inp.shape) == 4:  # Case 1: Input is RGB image
            inp = self.canonicalize(inp)

        canonical_cam2world = self.canonical_cam2world.repeat(inp.shape[0], 1, 1)
        canonical_intrinsics = self.canonical_intrinsics.repeat(inp.shape[0], 1, 1)
        frontalized_data = self.render(
            inp, canonical_cam2world, canonical_intrinsics, upsample=neural_upsample
        )

        return frontalized_data

    def forward(
        self,
        xs_data: Dict[str, torch.Tensor],
        xd_data: Dict[str, torch.Tensor]
    ):
        """
        Reenact the source image using driver(s).

        Parameters:
            - xs_data: The source's data. Must have 'image' key in it
            - xd_data: The driver' data. Must have 'image, 'cam2world', and 'intrinsics'
        """
        xs_triplane = self.canonicalize(xs_data['image'])

        # Frontalize xs
        with torch.no_grad():
            xs_face_frontal_data = self.frontalize(xs_data['image'], neural_upsample=True)
            xs_face_frontal_hr = (xs_face_frontal_data['image'] + 1) / 2  # Legacy issue

        with torch.no_grad():
            xd_triplane = self.canonicalize(xd_data['image'])
            xd_face_frontal_data = self.frontalize(xd_triplane, neural_upsample=False)
            xd_face_frontal_lr = F.interpolate(xd_face_frontal_data['image_raw'], (512, 512))
            xd_face_frontal_lr = (xd_face_frontal_lr + 1) / 2  # Legacy issue

        xs_triplane_newExp = self.exp_transfer(xs_face_frontal_hr, xd_face_frontal_lr, xs_triplane)
        driver_out = self.render(xs_triplane_newExp, xd_data['cam2world'], xd_data['intrinsics'])

        return driver_out
