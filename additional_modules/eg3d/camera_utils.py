import torch

import math

from rendering import math_utils


def create_cam2world_matrix(forward_vector, origin):
    """
    Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix.
    Works on batches of forward_vectors, origins. Assumes y-axis is up and that there is no camera roll.
    """

    forward_vector = math_utils.normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=origin.device).expand_as(forward_vector)

    right_vector = -math_utils.normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = math_utils.normalize_vecs(torch.cross(forward_vector, right_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), axis=-1)

    translation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin
    cam2world = (translation_matrix @ rotation_matrix)[:, :, :]
    assert(cam2world.shape[1:] == (4, 4))
    return cam2world


def FOV_to_intrinsics(fov_degrees, principal_x, principal_y):
    """
    Creates a 3x3 camera intrinsics matrix from the camera field of view, specified in degrees.
    Note the intrinsics are returned as normalized by image size, rather than in pixel units.
    Assumes principal point is at image center.
    """

    focal_length = (1 / (torch.tan(fov_degrees * 3.14159 / 360) * 1.414)).float()
    intrinsics = torch.eye(3, device=fov_degrees.device).unsqueeze(0).repeat(fov_degrees.shape[0], 1, 1)
    intrinsics[:, 0, 0] = focal_length
    intrinsics[:, 0, 2] = principal_x
    intrinsics[:, 1, 1] = focal_length
    intrinsics[:, 1, 2] = principal_y
    return intrinsics


class LookAtPoseSampler:
    """
    Same as GaussianCameraPoseSampler, except the
    camera is specified as looking at 'lookat_position', a 3-vector.

    Example:
    For a camera pose looking at the origin with the camera at position [0, 0, 1]:
    cam2world = LookAtPoseSampler.sample(math.pi/2, math.pi/2, torch.tensor([0, 0, 0]), radius=1)
    """

    @staticmethod
    def sample(
        horizontal_min, vertical_min, radius_mean,
        lookat_position,
        horizontal_max, vertical_max, radius_stddev=0,
        batch_size=1, device='cpu'
    ):
        if horizontal_max == -1:
            horizontal_max = horizontal_min
        h = torch.rand((batch_size, 1), device=device) * (horizontal_max - horizontal_min) + horizontal_min
        v = torch.rand((batch_size, 1), device=device) * (vertical_max - vertical_min) + vertical_min
        radius = torch.randn((batch_size, 1), device=device) * radius_stddev + radius_mean
        v = torch.clamp(v, 1e-5, math.pi - 1e-5)

        theta = h
        v = v / math.pi
        phi = torch.arccos(1 - 2*v)

        camera_origins = torch.zeros((batch_size, 3), device=device)

        camera_origins[:, 0: 1] = radius*torch.sin(phi) * torch.cos(math.pi-theta)
        camera_origins[:, 2: 3] = radius*torch.sin(phi) * torch.sin(math.pi-theta)
        camera_origins[:, 1: 2] = radius*torch.cos(phi)

        # forward_vectors = math_utils.normalize_vecs(-camera_origins)
        forward_vectors = math_utils.normalize_vecs(lookat_position - camera_origins)
        cam2world = create_cam2world_matrix(forward_vectors, camera_origins)

        return cam2world


class IntrinsicsSampler:
    @staticmethod
    def sample(
        focal_degrees_mean, principal_mean,
        focal_degrees_stddev=0, principal_stddev=0,
        batch_size=1, device='cpu'
    ):
        focal_degrees = torch.randn((batch_size), device=device) * focal_degrees_stddev + focal_degrees_mean
        principal_x = torch.randn((batch_size), device=device) * principal_stddev + principal_mean
        principal_y = torch.randn((batch_size), device=device) * principal_stddev + principal_mean

        return FOV_to_intrinsics(focal_degrees, principal_x, principal_y)

