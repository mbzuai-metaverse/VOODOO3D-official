import torch

import numpy as np
from argparse import Namespace

from additional_modules.deep3dfacerecon.util.load_mats import load_lm3d
from additional_modules.deep3dfacerecon.util.preprocess import align_img
from additional_modules.deep3dfacerecon.models.facerecon_model import FaceReconModel
from data_preprocessing.crop_image import CENTER_CROP_SIZE


class PoseEstimator:
    def __init__(self, device):
        self.opt = Namespace(**{
            'net_recon': 'resnet50',
            'phase': 'test',
            'init_path': None,
            'use_last_fc': False,
            'bfm_folder': 'additional_modules/deep3dfacerecon/BFM',
            'bfm_model': 'BFM_model_front.mat',
            'epoch': 20,
            'checkpoint_path': 'pretrained_models/deep3dfacerecon_epoch20.pth',
            'name': 'face_recon',
            'device': device,
            'camera_d': 10,
            'focal': 1015,
            'center': 112,
        })

        self.device = device
        self.lm3d_std = load_lm3d(self.opt.bfm_folder)

        self.pose_predictor = FaceReconModel(self.opt)
        self.pose_predictor.load_networks(self.opt.checkpoint_path)
        self.pose_predictor.eval()

    def _pose_est_process_data(self, im, lm):
        W, H = im.size
        lm = lm.copy().reshape([-1, 2])
        lm[:, -1] = H - 1 - lm[:, -1]
        _, im, lm, _, _, lm5p = align_img(im, lm, lm, self.lm3d_std)

        im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).to(self.device).unsqueeze(0)
        lm = torch.tensor(lm).to(self.device).unsqueeze(0)

        return im, lm

    @staticmethod
    def compute_rotation(angles):
        """
        Return:
            rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat

        Parameters:
            angles           -- torch.tensor, size (B, 3), radian
        """

        batch_size = angles.shape[0]
        ones = torch.ones([batch_size, 1])
        zeros = torch.zeros([batch_size, 1])
        x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],

        rot_x = torch.cat([
            ones, zeros, zeros,
            zeros, torch.cos(x), -torch.sin(x),
            zeros, torch.sin(x), torch.cos(x)
        ], dim=1).reshape([batch_size, 3, 3])

        rot_y = torch.cat([
            torch.cos(y), zeros, torch.sin(y),
            zeros, ones, zeros,
            -torch.sin(y), zeros, torch.cos(y)
        ], dim=1).reshape([batch_size, 3, 3])

        rot_z = torch.cat([
            torch.cos(z), -torch.sin(z), zeros,
            torch.sin(z), torch.cos(z), zeros,
            zeros, zeros, ones
        ], dim=1).reshape([batch_size, 3, 3])

        rot = rot_z @ rot_y @ rot_x
        return rot.permute(0, 2, 1)

    @torch.no_grad()
    def predict_pose(self, image, landmarks, batch_size=1):
        img_proc, lm_proc = self._pose_est_process_data(image, landmarks)
        data = {
            'imgs': img_proc,
            'lms': lm_proc,
        }
        self.pose_predictor.set_input(data)
        self.pose_predictor.test()

        pred_coeffs_dict_tensor = self.pose_predictor.pred_coeffs_dict
        pred_coeffs_dict = {k: v.cpu().numpy() for k, v in pred_coeffs_dict_tensor.items()}
        angle = pred_coeffs_dict['angle']
        R = self.compute_rotation(torch.from_numpy(angle))[0]

        # Extrinsics
        trans = pred_coeffs_dict['trans'][0]

        trans[2] += -10
        c = -np.dot(R, trans)
        c *= 0.27  # normalize camera radius
        c[1] += 0.006  # additional offset used in submission
        c[2] += 0.161  # additional offset used in submission
        radius = np.linalg.norm(c)
        c = c / radius * 2.7

        Rot = np.eye(3)
        Rot[0, 0] = 1
        Rot[1, 1] = -1
        Rot[2, 2] = -1
        R = np.dot(R, Rot)

        pose = np.eye(4)
        pose[0, 3] = c[0]
        pose[1, 3] = c[1]
        pose[2, 3] = c[2]
        pose[:3, :3] = R

        # Intrinsics
        focal = 2985.29 / CENTER_CROP_SIZE
        cx, cy = 0.5, 0.5
        K = np.eye(3)
        K[0][0] = focal
        K[1][1] = focal
        K[0][2] = cx
        K[1][2] = cy

        return K, pose


if __name__ == '__main__':
    from PIL import Image
    from face_detector import FaceDetector

    device = 'cuda'
    face_detector = FaceDetector(device)
    pose_estimator = PoseEstimator(device)
    img = Image.open('00000.png').convert('RGB')
    # with open('ffhq-dataset-v2.json', 'r') as f:
    #     gt_lm = np.array(json.load(f)['0']['in_the_wild']['face_landmarks'])

    lm, _ = face_detector(img)
    pose_estimator.predict_pose_batch([img], [lm])
    pose_estimator.debug()
