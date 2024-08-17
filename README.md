# EvalMono3D
This evaluation script computes the Average Precision (AP) for 3D bounding box predictions at an IoU threshold of 0.5. In contrast to KITTI, Waymo and NuScenes evaluation code, it accounts for full SO(3) rotation. Please find our project page here: 
[CARLA Drone: Monocular 3D Object Detection from a Different Perspective](https://meier-johannes94.github.io/CDrone/)

# Installation
```
conda create -n eval_tool python=3.9
conda activate eval_tool
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub  # for CUDA older than 11.7
conda install pytorch3d -c pytorch3d
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

For further details on PyTorch3D installation: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md


# Minimal example with a single image and cars
```
python eval.py \
    --name cdrone_test_example \
    --gt_ann example/cdrone_test_example.json \
    --pred_ann example/cdrone_test_example_pred.pth \
    --log_dir /tmp/eval_cdrone_test_example
```

# Source
The code is primarily based on PyTorch3D and [Cube R-CNN](https://github.com/facebookresearch/omni3d) and serves as a stand-alone evaluation script.
Therefore, this code is also licenced under [CC-BY-NC 4.0](https://github.com/facebookresearch/omni3d/blob/main/LICENSE.md).


# Citations
```
@article{meier2024cdrone,
  author    = {Meier, Johannes and Scalerandi, Luca and Dhaouadi, Oussema and Kaiser, Jacques and Araslanov Nikita and Cremers, Daniel},
  title     = {{CARLA Drone:} Monocular 3D Object Detection from a Different Perspective},
  journal   = {GCPR},
  year      = {2024},
}
```

```
@inproceedings{brazil2023omni3d,
  author =       {Garrick Brazil and Abhinav Kumar and Julian Straub and Nikhila Ravi and Justin Johnson and Georgia Gkioxari},
  title =        {{Omni3D}: A Large Benchmark and Model for {3D} Object Detection in the Wild},
  booktitle =    {CVPR},
  address =      {Vancouver, Canada},
  month =        {June},
  year =         {2023},
  organization = {IEEE},
}
```
