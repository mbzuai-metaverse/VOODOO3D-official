import torch
import torch.nn.functional as F

import numpy as np
import kornia.geometry.transform as K
from kornia.augmentation import ColorJiggle, AugmentationSequential
from kornia.augmentation import RandomChannelShuffle


class FaceAugmentor:
    def __init__(self):
        self.color_jitter = AugmentationSequential(
            ColorJiggle(0.25, 0.3, 0.3, 0.3, p=1.),
            RandomChannelShuffle(),
        )

    def _random_normal(self, size=(1,), trunc_val=2.5, rnd_state=None, device='cpu'):
        if rnd_state is None:
            rnd_state = np.random
        len = np.array(size).prod()
        result = np.empty((len,), dtype=np.float32)

        for i in range(len):
            while True:
                x = rnd_state.normal()
                if x >= -trunc_val and x <= trunc_val:
                    break
            result[i] = (x / trunc_val)

        return torch.from_numpy(result.reshape(size)).to(device)

    def _get_warp_params(self, batch_size, img_size, device):
        # random warp
        batch_cell_size = np.random.choice([img_size // (2**i) for i in range(1, 4)], batch_size)
        batch_cell_count = img_size // batch_cell_size + 1

        batch_grid_points = [
            torch.linspace(0, img_size, cell_count, device=device) for cell_count in batch_cell_count
        ]
        batch_mapx = [
            torch.broadcast_to(grid_points, (cell_count, cell_count)).clone()
            for grid_points, cell_count in zip(batch_grid_points, batch_cell_count)
        ]
        batch_mapy = [x.t() for x in batch_mapx]

        batch_mapx_resized = []
        batch_mapy_resized = []

        for cell_size, cell_count, mapx, mapy in zip(batch_cell_size, batch_cell_count, batch_mapx, batch_mapy):
            half_cell_size = cell_size // 2
            mapx[1:-1, 1:-1] = mapx[1:-1, 1:-1] +\
                self._random_normal(
                    size=(cell_count-2, cell_count-2),
                    device=device
                ) * (cell_size*0.24)
            mapy[1:-1, 1:-1] = mapy[1:-1, 1:-1] +\
                self._random_normal(
                    size=(cell_count-2, cell_count-2),
                    device=device
                ) * (cell_size*0.24)
            img_size = int(img_size)
            cell_size = int(cell_size)
            mapx = F.interpolate(mapx.unsqueeze(0).unsqueeze(0), (img_size + cell_size,) * 2, mode='bilinear')[
                :, 0, half_cell_size: -half_cell_size, half_cell_size: -half_cell_size
            ]
            mapy = F.interpolate(mapy.unsqueeze(0).unsqueeze(0), (img_size + cell_size,) * 2, mode='bilinear')[
                :, 0, half_cell_size: -half_cell_size, half_cell_size: -half_cell_size
            ]

            batch_mapx_resized.append(mapx)
            batch_mapy_resized.append(mapy)

        batch_mapx_resized = torch.cat(batch_mapx_resized)
        batch_mapy_resized = torch.cat(batch_mapy_resized)

        return batch_mapx_resized, batch_mapy_resized

    def _mask(self, faces, apply_rnd_mask):
        """
        Notice that this masking function is designed specifically for EG3D canonical space (yaw and pitch equal to 0).
        If you change the coordinate system, change this too.
        """
        B = faces.shape[0]
        N = faces.shape[-1]
        for i in range(B):
            mask_percent = 0.25
            mask_size = int(mask_percent * N)
            mask = torch.zeros_like(faces[i: i + 1])
            ones = torch.ones((1, 3, N - mask_size * 2, N - mask_size * 2), device=mask.device)
            mask[:, :, mask_size: N - mask_size, mask_size: N - mask_size] = ones
            faces[i: i + 1, ...] = faces[i: i + 1, ...] * mask

            if apply_rnd_mask:
                for _ in range(5):
                    # 32x32 patch masking
                    mask = torch.ones_like(faces[i: i + 1])
                    zeros = torch.zeros((1, 3, 64, 64), device=mask.device)
                    x = np.random.randint(int(N * mask_percent), N // 2)
                    y = np.random.randint(int(N * mask_percent), int(N * (1 - mask_percent)))
                    mask[:, :, x: x + zeros.shape[-2], y: y + zeros.shape[-1]] = zeros
                    faces[i: i + 1, ...] = faces[i: i + 1, ...] * mask - (1 - mask)

        return faces

    def _random_zoom_in(self, faces):
        size = faces.shape[-1]
        zoom_size_h = int(size * (0.7 + np.random.rand() * 0.3))
        zoom_size_w = int(size * (0.7 + np.random.rand() * 0.3))
        faces = K.center_crop(faces, (zoom_size_h, zoom_size_w))
        faces = K.resize(faces, (size, size))

        return faces

    def _random_zoom_out(self, faces):
        size = faces.shape[-1]
        pad_h = np.random.randint(int(size * 0.2))
        pad_w = np.random.randint(int(size * 0.2))
        faces = F.pad(faces, (pad_h, pad_w), mode='constant')
        faces = K.resize(faces, (size, size))

        return faces
    
    def _random_color_patch(self, faces):
        mask_percent = 0.25
        B = faces.shape[0]
        N = faces.shape[-1]

        aug_faces = self.color_jitter(faces)

        for i in range(B):
            for _ in range(20):
                x = np.random.randint(int(N * mask_percent), int(N * (1 - mask_percent)))
                y = np.random.randint(int(N * mask_percent), int(N * (1 - mask_percent)))

                subfaces = faces[i, :, x: x + 128, y: y + 128]
                subfaces = self.color_jitter(subfaces)

                aug_faces[i, :, x: x + 128, y: y + 128] = subfaces
        return aug_faces

    @torch.no_grad()
    def __call__(self, faces, target_size, apply_color_aug=True, apply_rnd_mask=True, apply_rnd_zoom=True):
        if target_size is not None:
            faces = F.interpolate(faces, size=target_size)

        if apply_color_aug:
            faces = self._random_color_patch(faces)

        zoom_type = 2 if not apply_rnd_zoom else np.random.randint(3)
        if zoom_type == 0:
            faces = self._random_zoom_in(faces)
        elif zoom_type == 1:
            faces = self._random_zoom_out(faces)

        faces = self._mask(faces, apply_rnd_mask)

        return faces
