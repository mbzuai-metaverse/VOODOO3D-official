import torch
import torch.nn as nn

import numpy as np

from additional_modules.eg3d.networks import TriPlaneGenerator
from additional_modules.eg3d.camera_utils import LookAtPoseSampler, IntrinsicsSampler


NETWORK_PKL = 'experiments/pretrained_models/eg3d_ffhq_rebalance.pth'


class EG3DSampler(nn.Module):
    def __init__(self):
        super().__init__()

        eg3d_data = torch.load(NETWORK_PKL, map_location='cpu')
        G = TriPlaneGenerator(**eg3d_data['init_kwargs'])
        G.init_kwargs = eg3d_data['init_kwargs']
        G.load_state_dict(eg3d_data['state_dict'])
        G.neural_rendering_resolution = eg3d_data['neural_rendering_resolution']
        G.rendering_kwargs = eg3d_data['rendering_kwargs']
        G.rendering_kwargs['ray_start'] = 2.0
        G.rendering_kwargs['ray_end'] = 3.5
        G.rendering_kwargs['depth_resolution'] = 52
        G.rendering_kwargs['depth_resolution_importance'] = 60
        self.G = G

        self.pose_sampler = LookAtPoseSampler()
        self.intrinsics_sampler = IntrinsicsSampler()

        self.register_buffer('lookat_position', torch.tensor([0, 0, 0]))

    def render(self, z, yaw, pitch):
        device = self.lookat_position.device
        lookat_position = self.lookat_position.unsqueeze(0)

        cam2world_pose = self.pose_sampler.sample(
            yaw, pitch, 2.7,
            lookat_position,
            yaw, pitch, 0.0,
            batch_size=1, device=device
        )
        intrinsics = self.intrinsics_sampler.sample(
            18.837, 0.5,
            0.0, 0.0,
            batch_size=1, device=device
        )

        radius = torch.linalg.vector_norm(cam2world_pose[:, :3, 3], dim=1, keepdim=True)
        conditioning_cam2world_pose = self.pose_sampler.sample(
            np.pi/2, np.pi/2, radius,
            lookat_position,
            np.pi/2, np.pi/2, 0,
            batch_size=1, device=device
        )

        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

        ws = self.G.mapping(z, conditioning_params, truncation_psi=0.7, truncation_cutoff=14)
        out = self.G.synthesis(ws, camera_params)

        return out['image']

    @torch.no_grad()
    def forward(self, num_views, batch_size, z=None):
        device = self.lookat_position.device
        lookat_position = self.lookat_position.unsqueeze(0).repeat(batch_size, 1)

        if z is None:
            z = torch.randn((batch_size, self.G.z_dim), device=device)
        assert z.shape[0] == batch_size

        all_out = []
        for view_idx in range(num_views):
            cam2world_pose = self.pose_sampler.sample(
                0.71, 1.11, 2.7,
                lookat_position,
                2.42, 2.02, 0.1,
                batch_size=batch_size, device=device
            )
            intrinsics = self.intrinsics_sampler.sample(
                18.837, 0.5,
                1.5, 0.02,
                batch_size=batch_size, device=device
            )

            radius = torch.linalg.vector_norm(cam2world_pose[:, :3, 3], dim=1, keepdim=True)
            conditioning_cam2world_pose = self.pose_sampler.sample(
                np.pi/2, np.pi/2, radius,
                lookat_position,
                np.pi/2, np.pi/2, 0,
                batch_size=batch_size, device=device
            )

            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
            conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

            ws = self.G.mapping(z, conditioning_params, truncation_psi=0.7, truncation_cutoff=14)
            out = self.G.synthesis(ws, camera_params)
            out['cam2world'] = cam2world_pose
            out['intrinsics'] = intrinsics

            all_out.append(out)

        return all_out
