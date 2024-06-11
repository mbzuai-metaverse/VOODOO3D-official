import torch.nn as nn


from .encoder import resnet34
from .decoder import DeepLabV3Decoder


class DeepLabV3(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        self.encoder = resnet34(input_channels=input_channels)
        self.decoder = DeepLabV3Decoder(in_channels=128)

    def forward(self, x):
        feat = self.encoder(x)
        out = self.decoder(*feat)

        return out
