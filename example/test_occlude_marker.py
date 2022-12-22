#!/usr/bin/env python3

import boxx
from boxx import os, glob


def imread(path):
    img = cv2.imread(path)
    img[1000:1500, 1000:1500] = 124
    return img


boxx.imread = imread

from calibrating import *
from calibrating import Cam, Stereo

if __name__ == "__main__":
    from boxx import *

    enable_cache = False
    enable_cache = True
    root = os.path.abspath(
        os.path.join(
            __file__,
            "../../../calibrating_example_data/paired_stereo_and_depth_cams_aruco",
        )
    )
    board = PredifinedArucoBoard1(occlusion=True)
    caml = Cam(
        glob(os.path.join(root, "*", "stereo_l.jpg")),
        board,
        name="caml",
        enable_cache=enable_cache,
    )
    camr = Cam(
        glob(os.path.join(root, "*", "stereo_r.jpg")),
        board,
        name="camr",
        enable_cache=enable_cache,
    )

    stereo = Stereo(caml, camr)
    shows - stereo.vis()
    print(stereo)

    key = sorted(caml)[0]
    img = imread(caml[key]["path"])
    T = caml[key]["T"]
    re = caml.get_calibration_board_T(img)
    T_board_in_cam = re["T"]

    assert not (T - T_board_in_cam).round(9).any()


if 0:
    camd = Cam(
        glob(os.path.join(root, "*", "depth_cam_color.jpg")),
        board,
        name="camd",
        undistorted=True,
        enable_cache=enable_cache,
    )
