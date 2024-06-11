import torch.nn as nn

from additional_modules.segformer.backbone import mit_b0
from additional_modules.segformer.segformer_head import SegFormerHead

from additional_modules.segformer.base_model import BaseSegmentor


class EncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 neck=None,
                 num_classes=256,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(EncoderDecoder, self).__init__()

        self.num_classes = num_classes
        self.backbone = mit_b0()
        self._init_decode_head()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

        assert self.with_decode_head

    def _init_decode_head(self):
        """Initialize ``decode_head``"""
        self.decode_head = SegFormerHead(
            in_channels=[32, 64, 160, 256],
            in_index=[0, 1, 2, 3],
            feature_strides=[4, 8, 16, 32],
            channels=128,
            dropout_ratio=0.1,
            num_classes=self.num_classes,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            align_corners=False,
            decoder_params=dict(embed_dim=256),
        )
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        super(EncoderDecoder, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward(self, img):
        """Forward function for training.

        Args:
            img (Tensor): Input images.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(img)
        x = self.decode_head(x)

        return x
