import torch
import torch.nn as nn

from typing import List, Dict

from additional_modules.deeplabv3.deeplabv3 import DeepLabV3
from additional_modules.segformer.backbone import Block, OverlapPatchEmbed
from additional_modules.eg3d.networks import OSGDecoder
from models import get_model
from rendering.ray_sampler import RaySampler
from rendering.triplane_rendering.renderer import ImportanceRenderer
from utils.registry import MODEL_REGISTRY


class PositionalEncoder(nn.Module):
    def __init__(self, img_size: int):
        super().__init__()

        h_linspace = torch.linspace(-1, 1, img_size)
        w_linspace = torch.linspace(-1, 1, img_size)
        gh, gw = torch.meshgrid(h_linspace, w_linspace, indexing='xy')
        gh, gw = gh.unsqueeze(0), gw.unsqueeze(0)
        id_grid = torch.cat((gh, gw), dim=0).unsqueeze(0)
        self.register_buffer('id_grid', id_grid)

    def _add_positional_encoding(self, img):
        id_grid = self.id_grid.repeat(img.shape[0], 1, 1, 1)
        x = torch.cat((img, id_grid), dim=1)

        return x


class ELow(PositionalEncoder):
    def __init__(self, img_size: int = 512, img_channels: int = 3):
        super().__init__(img_size)

        self.deeplabv3_backbone = DeepLabV3(input_channels=img_channels + 2)
        self.patch_embed = OverlapPatchEmbed(
            img_size=img_size // 8, patch_size=3, stride=2, in_chans=256, embed_dim=1024
        )

        self.block1 = Block(dim=1024, num_heads=4, mlp_ratio=2, sr_ratio=1)
        self.block2 = Block(dim=1024, num_heads=4, mlp_ratio=2, sr_ratio=1)
        self.block3 = Block(dim=1024, num_heads=4, mlp_ratio=2, sr_ratio=1)
        self.block4 = Block(dim=1024, num_heads=4, mlp_ratio=2, sr_ratio=1)
        self.block5 = Block(dim=1024, num_heads=4, mlp_ratio=2, sr_ratio=1)

        self.up1 = nn.PixelShuffle(upscale_factor=2)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv1 = nn.Conv2d(256, 128, 3, 1, 1, bias=True)
        self.act1 = nn.ReLU()
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=True)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(128, 96, 3, 1, 1, bias=True)

    def forward(self, img: torch.Tensor):
        x = self._add_positional_encoding(img)

        x = self.deeplabv3_backbone(x)
        x, H, W = self.patch_embed(x)

        x = self.block1(x, H, W)
        x = self.block2(x, H, W)
        x = self.block3(x, H, W)
        x = self.block4(x, H, W)
        x = self.block5(x, H, W)

        x = x.reshape(img.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = self.up1(x)
        x = self.up2(x)

        x = self.conv1(x)
        x = self.act1(x)
        x = self.up3(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)

        return x


class EHigh(PositionalEncoder):
    def __init__(self, img_size: int = 512, img_channels: int = 3):
        super().__init__(img_size)

        self.conv1 = nn.Conv2d(img_channels + 2, 64, 7, 2, 3, bias=True)
        self.act1 = nn.LeakyReLU(0.01)

        self.conv2 = nn.Conv2d(64, 96, 3, 1, 1, bias=True)
        self.act2 = nn.LeakyReLU(0.01)

        self.conv3 = nn.Conv2d(96, 96, 3, 1, 1, bias=True)
        self.act3 = nn.LeakyReLU(0.01)

        self.conv4 = nn.Conv2d(96, 96, 3, 1, 1, bias=True)
        self.act4 = nn.LeakyReLU(0.01)

        self.conv5 = nn.Conv2d(96, 96, 3, 1, 1, bias=True)
        self.act5 = nn.LeakyReLU(0.01)

    def forward(self, img: torch.Tensor):
        x = self._add_positional_encoding(img)
        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.act3(x)

        x = self.conv4(x)
        x = self.act4(x)

        x = self.conv5(x)
        x = self.act5(x)

        return x


class Lp3DEncoder(nn.Module):
    def __init__(self, img_size: int = 512, img_channels: int = 3, triplane_nd: int = 32):
        super().__init__()
        self.img_size = img_size

        self.elo = ELow(img_size, img_channels)
        self.ehi = EHigh(img_size, img_channels)

        self.conv1 = nn.Conv2d(192, 256, 3, 1, 1, bias=True)
        self.act1 = nn.LeakyReLU(0.01)

        self.conv2 = nn.Conv2d(256, 128, 3, 1, 1, bias=True)
        self.act2 = nn.LeakyReLU(0.01)

        self.patch_embed = OverlapPatchEmbed(
            img_size=img_size // 2, patch_size=3, stride=2, in_chans=128, embed_dim=1024
        )
        self.transformer_block = Block(dim=1024, num_heads=2, mlp_ratio=2, sr_ratio=2)

        self.up = nn.PixelShuffle(upscale_factor=2)

        self.conv3 = nn.Conv2d(352, 256, 3, 1, 1, bias=True)
        self.act3 = nn.LeakyReLU(0.01)

        self.conv4 = nn.Conv2d(256, 128, 3, 1, 1, bias=True)
        self.act4 = nn.LeakyReLU(0.01)

        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1, bias=True)
        self.act5 = nn.LeakyReLU(0.01)

        self.conv6 = nn.Conv2d(128, triplane_nd * 3, 3, 1, 1, bias=True)

    def forward(self, img: torch.Tensor):
        assert img.shape[-1] == self.img_size and img.shape[-2] == self.img_size

        f_lo = self.elo(img)
        f_hi = self.ehi(img)

        f = torch.cat((f_lo, f_hi), dim=1)
        f = self.conv1(f)
        f = self.act1(f)

        f = self.conv2(f)
        f = self.act2(f)

        f, H, W = self.patch_embed(f)
        f = self.transformer_block(f, H, W)
        f = f.reshape(img.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()

        f = self.up(f)
        f = torch.cat((f, f_lo), dim=1)

        f = self.conv3(f)
        f = self.act3(f)

        f = self.conv4(f)
        f = self.act4(f)

        f = self.conv5(f)
        f = self.act5(f)

        f = self.conv6(f)

        return f


@MODEL_REGISTRY.register()
class Lp3D(nn.Module):
    def __init__(
        self,
        neural_rendering_resolution: int,  # Render at this resolution and use superres to upsample to 512x512
        triplane_nd: int,  # Triplane's number of channels
        triplane_h: int,  # Triplane height
        triplane_w: int,  # Triplane width
        rendering_kwargs,
        superresolution_kwargs,
    ):
        super().__init__()

        self.triplane_nd = triplane_nd
        self.triplane_h = triplane_h
        self.triplane_w = triplane_w
        self.neural_rendering_resolution = neural_rendering_resolution
        self.superresolution_opt = superresolution_kwargs
        self.rendering_kwargs = rendering_kwargs

        self._setup_modules()

    def _setup_modules(self):
        # For now only support 512x512 input image and 256x256 triplane
        self.triplane_encoder = Lp3DEncoder(triplane_nd=self.triplane_nd)

        self.renderer = ImportanceRenderer()
        self.decoder = OSGDecoder(
            self.triplane_nd,
            {
                'decoder_lr_mul': self.rendering_kwargs.get('decoder_lr_mul', 1),
                'decoder_output_dim': self.triplane_nd
            }
        )
        self.ray_sampler = RaySampler()
        self.superresolution = get_model(self.superresolution_opt)

    def render(self, planes, cam2world, intrinsics, upsample=True):
        """
        Render the triplane using cam2wolrd and intrinsics matrices

        Parameters:
            - triplane (Tensor)
            - cam2world (Tensor)
            - intrinsics (Tensor)
        Returns:
            - image (Tensor) [Range -1..1]: The rendered images
            - cam2world (Tensor): The input cam2world. Can be useful to stack multiple renderings.
            - intrinsics (Tensor): The input intrinsics. Can be useful to stack multiple renderings.
        """
        ray_origins, ray_directions = self.ray_sampler(
            cam2world, intrinsics, self.neural_rendering_resolution
        )

        batch_size = cam2world.shape[0]

        feature_samples, depth_samples, _, _ = self.renderer(
            planes,
            self.decoder,
            ray_origins,
            ray_directions,
            self.rendering_kwargs
        )  # channels last

        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(batch_size, feature_samples.shape[-1], H, W).contiguous()
        rgb_image_raw = feature_image[:, :3]

        if upsample:
            rgb_image = self.superresolution(rgb_image_raw, feature_image)
        else:
            rgb_image = None
       
        depth_image = depth_samples.permute(0, 2, 1).reshape(batch_size, 1, H, W).contiguous()

        return {
            'image_raw': rgb_image_raw,
            'image': rgb_image,
            'planes': planes,
            'depth': depth_image,
            'feature_image': feature_image,
            'cam2world': cam2world,
            'intrinsics': intrinsics
        }

    def canonicalize(self, image: torch.Tensor):
        """
        Transform the input image to the canonicalized 3D space which represented by a triplane

        Parameters:
            - image (Tensor): Input image
        Returns:
            - triplanes (Tensor): The canonical representation of the input
        """

        image = (image + 1) / 2.  # Legacy issue :(
        triplanes = self.triplane_encoder(image)
        B = triplanes.shape[0]
        triplanes = triplanes.view(B, 3, 32, triplanes.shape[-2], triplanes.shape[-1]).contiguous()

        return triplanes

    def forward(
        self,
        xs_data: Dict[str, torch.Tensor],
        all_xds_data: List[Dict[str, torch.Tensor]]
    ):
        """ 
        Render the source image using camera parameters from the driver(s).

        This inference function support multiple camera inputs. Can be useful when
        training in which the loss is calculated on multiple views of a single
        source image

        Parameters:
            - xs_data: The source's data. Must have 'image' key in it
            - all_xds_data: All drivers' data. Each of them must have 'cam2world' and 'intrinsics'
        """
        xs_triplane = self.canonicalize(xs_data['image'])

        all_out = []
        for xd_data in all_xds_data:
            driver_out = self.render(xs_triplane, xd_data['cam2world'], xd_data['intrinsics'])
            all_out.append(driver_out)

        return all_out
