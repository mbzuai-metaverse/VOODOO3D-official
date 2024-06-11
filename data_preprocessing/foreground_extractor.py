import torch

import numpy as np

from additional_modules.modnet.modnet import MODNet


PRETRAINED_PATH = 'pretrained_models/modnet_photographic_portrait_matting.ckpt'


class ForegroundExtractor:
    def __init__(self, device):
        self.fg_extractor = MODNet().eval().to(device)

        state_dict = torch.load(PRETRAINED_PATH, map_location='cpu')
        state_dict = {x.replace('module.', '', 1): y for x, y in state_dict.items()}
        self.fg_extractor.load_state_dict(state_dict)

        self.device = device

    @torch.no_grad()
    def __call__(self, img):
        img = np.transpose(np.array(img), (2, 0, 1))[None, ...] / 255.
        img = torch.from_numpy(img).to(self.device).float()
        matte = self.fg_extractor(img)
        matte = np.transpose(matte.cpu().numpy()[0], (1, 2, 0))

        return matte
