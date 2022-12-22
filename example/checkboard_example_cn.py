#!/usr/bin/env python3

import os
from glob import glob
import calibrating  # pip install calibrating
from calibrating import imread, shows

checkboard_img_dir = os.path.abspath(
    os.path.join(
        __file__,
        "../../../calibrating_example_data/paired_stereo_and_depth_cams_checkboard",
    )
)

assert os.path.isdir(
    checkboard_img_dir
), 'Not found "calibrating_example_data", please "git clone https://github.com/yl-data/calibrating_example_data"'

# ▮ 标定所有相机的内参
# 准备棋盘格特征提取器
board = calibrating.Chessboard(checkboard=(7, 10), size_mm=22.564)

# 实例化的过程中会自动标定内参 cam.K, cam.D, 并在 `/tmp/calibrating-*` 下存放 feature 可视化
# 左目/右目/depth相机
caml = calibrating.Cam(glob(f"{checkboard_img_dir}/*/stereo_l.jpg"), board)
camr = calibrating.Cam(glob(f"{checkboard_img_dir}/*/stereo_r.jpg"), board)
camd = calibrating.Cam(glob(f"{checkboard_img_dir}/*/depth_cam_color.jpg"), board)
built_in_intrinsics = dict(
    fx=1474.1182177692722,
    fy=1474.125874583105,
    cx=1037.599716850734,
    cy=758.3072639103259,
)
# depth 需要与深度相机的内置的内参成对使用
camd.load(built_in_intrinsics)

print(caml, camr, camd)
# 用浏览器可视化在相机视野中每一块标定板的 image_point 和 Rt
shows([cam.vis_image_points_cover() for cam in (caml, camr, camd)])

# ▮ 标定双目相机
stereo = calibrating.Stereo(caml, camr)
print(stereo)
# 左右图像 rectify 对齐可视化
caml.vis_stereo(camr, stereo)
# depth 精度理论分析
# stereo.precision_analysis()

# ▮ 使用 stereo_matching 算法(SemiGlobalBlockMatching) 来计算深度
max_depth = 3.5
stereo_matching = calibrating.SemiGlobalBlockMatching()
stereo.set_stereo_matching(stereo_matching, max_depth=max_depth)
# 准备一对左右相机的图像
key = caml.valid_keys_intersection(camr)[0]
imgl = imread(caml[key]["path"])
imgr = imread(camr[key]["path"])
stereo_result = stereo.get_depth(imgl, imgr)
depth_stereo = stereo_result["unrectify_depth"]
# 1~3.5m 内的绝对深度可视化
depth_stereo_vis = calibrating.vis_depth(depth_stereo, fix_range=(1, max_depth))
undistort_img1 = stereo_result["undistort_img1"]
# 注意: stereo_matching 可以替换为基于深度学习的算法

# ▮ 重投影深度相机的 depth 到左相机 caml
# depth_cam 在左目坐标系下的外参 (4x4)
T_d_in_l = caml.get_T_cam2_in_self(camd)

# 读取 depth_cam 的 depth, 单位是米
depthd = imread(camd[key]["path"].replace("color.jpg", "depth.png")) / 1000

# 把 depth_cam 的 depth 重投影到左目的 depth (会自动对 depth 插值)
depth_reproject = caml.project_cam2_depth(camd, depthd, T_d_in_l)
depth_reproject_vis = calibrating.vis_depth(depth_reproject, fix_range=(1, max_depth))

# 浏览器中可视化图像, 注意:
#   - 所有的深度都是和去畸变后的左相机图像对齐
#   - example 的标定图像只有 16 张, 所以标定的精度不高, 实际使用中, 标定图像最好超过 30 张
shows([depth_stereo_vis, undistort_img1, depth_reproject_vis])
