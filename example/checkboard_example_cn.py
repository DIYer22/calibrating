#!/usr/bin/env python3

import os
from glob import glob
from boxx import imread
import calibrating  # pip install calibrating

checkboard_img_dir = os.path.abspath(
    os.path.join(
        __file__,
        "../../../calibrating_example_data/paired_stereo_and_depth_cams_checkboard",
    )
)

assert os.path.isdir(
    checkboard_img_dir
), 'Not found "calibrating_example_data", please "git clone https://github.com/yl-data/calibrating_example_data"'

# 准备棋盘格特征提取器
feature_lib = calibrating.CheckboardFeatureLib(checkboard=(7, 10), size_mm=22.564)

# 实例化的过程中会自动标定内参 cam.K, cam.D, 并在 /tmp 下存放 feature可视化
# 左目/右目/depth相机
caml = calibrating.Cam(glob(f"{checkboard_img_dir}/*/stereo_l.jpg"), feature_lib)
camr = calibrating.Cam(glob(f"{checkboard_img_dir}/*/stereo_r.jpg"), feature_lib)
camd = calibrating.Cam(glob(f"{checkboard_img_dir}/*/depth_cam_color.jpg"), feature_lib)

# 双目标定与 rectify 可视化
stereo = caml.stereo_with(camr)
caml.vis_stereo(camr, stereo)

# depth_cam 在左目坐标系下的外参 (4x4)
T_d_in_l = caml.get_T_cam2_in_self(camd)

# 读取左目图像和 depth_cam 的 depth 用于可视化
key = caml.valid_keys_intersection(camd)[0]
imgl = imread(caml[key]["path"])
depthd = imread(camd[key]["path"].replace("color.jpg", "depth.png"))

# 把 depth_cam 的 depth 重投影到左目的 depth (会自动对 depth 插值)
depthl = caml.project_cam2_depth(camd, depthd, T_d_in_l)

# 动态可视化 depth 重投影对齐效果
caml.vis_project_align(imgl, depthl)
