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

# Prepare checkerboard feature extractor
feature_lib = calibrating.CheckboardFeatureLib(checkboard=(7, 10), size_mm=22.564)

# In the process of instantiation, the internal parameters cam.K, cam.D will be automatically calibrated, and feature visualization will be stored under /tmp
# left_cam/right_cam/depth_cam
caml = calibrating.Cam(glob(f"{checkboard_img_dir}/*/stereo_l.jpg"), feature_lib)
camr = calibrating.Cam(glob(f"{checkboard_img_dir}/*/stereo_r.jpg"), feature_lib)
camd = calibrating.Cam(glob(f"{checkboard_img_dir}/*/depth_cam_color.jpg"), feature_lib)

# Calibrate stereo cameras and visualization rectify effect
stereo = caml.stereo_with(camr)
caml.vis_stereo(camr, stereo)

# External parameters of depth_cam in the left_cam coordinate system (4x4)
T_d_in_l = caml.get_T_cam2_in_self(camd)

# Prepare the left_cam's image and depth_cam's depth for visualization
key = caml.valid_keys_intersection(camd)[0]
imgl = imread(caml[key]["path"])
depthd = imread(camd[key]["path"].replace("color.jpg", "depth.png"))

# Reproject the depth of depth_cam to the left_cam's depth (the depth will be automatically interpolated)
depthl = caml.project_cam2_depth(camd, depthd, T_d_in_l)

# Dynamic visualization depth reprojection alignment effect
caml.vis_project_align(imgl, depthl)
