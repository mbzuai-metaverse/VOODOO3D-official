# [CVPR 2024] VOODOO 3D: <ins>Vo</ins>lumetric P<ins>o</ins>rtrait <ins>D</ins>isentanglement f<ins>o</ins>r <ins>O</ins>ne-Shot 3D Head Reenactment

[![arXiv](https://img.shields.io/badge/arXiv-2312.04651-red?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2312.04651)
[![youtube](https://img.shields.io/badge/video-Youtube-white?logo=youtube&logoColor=red)](https://arxiv.org/abs/2312.04651)
[![homepage](https://img.shields.io/badge/project-Homepage-orange?logo=Homepage&logoColor=orange)](https://arxiv.org/abs/2312.04651)
[![LICENSE](https://img.shields.io/badge/license-MIT%202.0-blue?logo=C&logoColor=blue)](https://github.com/MBZUAI-Metaverse/VOODOO3D-official/LICENSE)

![teaser](./resources/github_readme/teaser.gif)

## Overview
This is the official implementation of VOODOO 3D: a high-fidelity 3D-aware one-shot head reenactment technique. Our method transfers the expression of a driver to a source and produces view consistent renderings for holographic displays.

For more details of the method and experimental results of the project, please checkout our [paper](https://arxiv.org/abs/2312.04651), [youtube video](https://www.youtube.com/watch?v=Gu3oPG0_BaE), or the [project page](https://p0lyfish.github.io/voodoo3d/).

## Installation
First, clone the project:
```
git clone https://github.com/MBZUAI-Metaverse/VOODOO3D-official
```
The implementation only requires standard libraries. You can install all the dependencies using pip:
```
conda create -n voodoo3d python=3.10  # optional
pip install -r requirements.txt
```

You need to download the BFM model for the pose estimation. You can download it [here](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/the_tran_mbzuai_ac_ae/EasQUk8MESRMtIphdDA7T14BDzj83frLGU3VQoWM6CG6iQ?e=C4vZ0k) and put it into `./additional_modules/deep3dfacerecon`.

Next, prepare the pretrained weights and put them into `./pretrained_models`. You can download them [here](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/the_tran_mbzuai_ac_ae/EUWFHRIXZrxEo2Ak2zKhxfwBzmaLFjnBLmmi-5BoTuPU4w?e=m70Ly5).

## Inference
### 3D Head Reenactment
Use the following command to test the model:
```
python test_voodoo3d.py --source_root <IMAGE_FOLDERS / IMAGE_PATH> \
                    --driver_root <IMAGE_FOLDERS / IMAGE_PATH> \
                    --config_path configs/voodoo3d.yml \
                    --model_path pretrained_models/voodoo3d.pth \
                    --save_root <SAVE_ROOT> \
```
Where `source_root` and `driver_root` are either image folders or image paths of the sources and drivers respectively. `save_root` is the folder root that you want to save the results. This script will generate pairwise reenactment results of the sources and drivers in the input folders / paths. For example, to test with our provided images:
```
python test_voodoo3d.py --source_root resources/images/sources \
                    --driver_root resources/images/drivers \
                    --config_path configs/voodoo3d.yml \
                    --model_path pretrained_models/voodoo3d.pth \
                    --save_root results/voodoo3d_test \
```
### Fine-tuned Lp3D for 3D Reconstruction
[Lp3D](https://research.nvidia.com/labs/nxp/lp3d/) is the state-of-the-art 3D Portrait Reconstruction model. As mentioned in the VOODOO 3D paper, we had a reimplementation of this model but fine-tuned on in-the-wild data. To evaluate this model, use the following script:
```
python test_lp3d.py --source_root <IMAGE_FOLDERS / IMAGE_PATH> \
                    --config_path configs/lp3d.yml \
                    --model_path pretrained_models/voodoo3d.pth \
                    --save_root <SAVE_ROOT> \
                    --cam_batch_size <BATCH_SIZE>
```
where `source_root` is either an image folder or an image path of the images that will be reconstructed in 3D. `SAVE_ROOT` is the destination of the results. `BATCH_SIZE` is the testing batch size (the higher, the faster). For each image in the input folder, the model will generate a rendered video of its corresponding 3D head using a fixed camera trajectory. Here is an example using our provided images:
```
python test_lp3d.py --source_root resources/images/sources \
                    --config_path configs/lp3d.yml \
                    --model_path pretrained_models/voodoo3d.pth \
                    --save_root results/lp3d_test \
                    --cam_batch_size 2
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
This work would not be possible without the following projects:

- [eg3d](https://github.com/NVlabs/eg3d): We borrowed the data preprocessing and the generative model code to synthesize the data during training.
- [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch): We borrowed this code to predict the camera pose and process the data.
- [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch): We borrowed DeepLabV3 implementation from this project.
- [MODNet](https://github.com/ZHKKKe/MODNet): We borrowed the foreground extraction code from this project.
- [SegFormer](https://github.com/NVlabs/SegFormer): We borrowed the transformer blocks from this project.
- [GFPGAN](https://github.com/TencentARC/GFPGAN): We use GFPGAN as our super-resolution module

If you see your code used in this implementation but haven't properly acknowledged, please contact me via [tranthephong33@gmail.com](tranthephong33@gmail.com).

## BibTeX
If our code is useful for your research or application, please cite our paper:
```
@inproceedings{tran2023voodoo,
	title = {VOODOO 3D: Volumetric Portrait Disentanglement for One-Shot 3D Head Reenactment},
	author = {Tran, Phong and Zakharov, Egor and Ho, Long-Nhat and Tran, Anh Tuan and Hu, Liwen and Li, Hao},
	year = 2024,
	booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition}
}
```

## Contact
For any questions or issues, please open an issue or contact [tranthephong33@gmail.com](mailto:tranthephong33@gmail.com).
