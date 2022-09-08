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

# ▮ Calibrate the intrinsic of all cameras
# Prepare checkerboard feature extractor
feature_lib = calibrating.CheckboardFeatureLib(checkboard=(7, 10), size_mm=22.564)

# In the process of instantiation, the intrinsic (cam.K, cam.D) will be automatically calibrated, and feature visualization will be stored under `/tmp/calibrating-*`
# left_cam/right_cam/depth_cam
caml = calibrating.Cam(glob(f"{checkboard_img_dir}/*/stereo_l.jpg"), feature_lib)
camr = calibrating.Cam(glob(f"{checkboard_img_dir}/*/stereo_r.jpg"), feature_lib)
camd = calibrating.Cam(glob(f"{checkboard_img_dir}/*/depth_cam_color.jpg"), feature_lib)
print(caml, camr, camd)
# Using the browser to visualize the image_point and Rt of each calibration board in the camera's field of view
shows([cam.vis_image_points_cover() for cam in (caml, camr, camd)])

# ▮ Calibrate the stereo camera
stereo = calibrating.Stereo(caml, camr)
print(stereo)
# Visualization rectify effect
caml.vis_stereo(camr, stereo)
# Theoretical analysis of depth precision
# stereo.precision_analysis()

# ▮ Use the stereo_matching algorithm (by SemiGlobalBlockMatching) to calculate the depth
max_depth = 3.5
stereo_matching = calibrating.SemiGlobalBlockMatching()
stereo.set_stereo_matching(stereo_matching, max_depth=max_depth)

key = caml.valid_keys_intersection(camr)[0]
imgl = imread(caml[key]["path"])
imgr = imread(camr[key]["path"])
stereo_result = stereo.get_depth(imgl, imgr)
depth_stereo = stereo_result["unrectify_depth"]
# Absolute depth visualization within 1~3.5m
depth_stereo_vis = calibrating.vis_depth(depth_stereo, fix_range=(1, max_depth))
undistort_img1 = stereo_result["undistort_img1"]
# Note: stereo_matching can be replaced with deep learning based algorithms

# ▮ Reproject depth_cam's depth to left_cam
# Extristric of depth_cam in the left_cam's coordinate system (4x4)
T_d_in_l = caml.get_T_cam2_in_self(camd)

# Prepare the depth_cam's depth (unit is meter)
depthd = imread(camd[key]["path"].replace("color.jpg", "depth.png")) / 1000

# Reproject the depth of depth_cam as left_cam's depth (the depth will be automatically interpolated)
depth_reproject = caml.project_cam2_depth(camd, depthd, T_d_in_l)
depth_reproject_vis = calibrating.vis_depth(depth_reproject, fix_range=(1, max_depth))

# Visualize the image in the browser, note:
#   - all depths are aligned with the undistorted left camera image
#   - There are only 16 calibration images in this example, so the calibration accuracy is not high. In actual use, the calibration images should preferably exceed 30
shows([depth_stereo_vis, undistort_img1, depth_reproject_vis])
