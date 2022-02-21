#!/usr/bin/env python3

import cv2
import boxx
from boxx import os, glob, imread, np
from calibrating import *
from calibrating import Cam, SemiGlobalBlockMatching, Stereo, vis_depth, get_test_cams


if __name__ == "__main__":
    from boxx import *

    feature_type = "checkboard"
    feature_type = "aruco"
    cams = get_test_cams(feature_type)
    caml, camr, camd = cams["caml"], cams["camr"], cams["camd"]

    key = caml.valid_keys_intersection(camd)[0]
    imgl = imread(caml[key]["path"])
    imgr = imread(camr[key]["path"])
    color_path_d = camd[key]["path"]
    depthd = imread(color_path_d.replace("color.", "depth.").replace(".jpg", ".png"))
    undistort_imgl = cv2.undistort(imgl, caml.K, caml.D)

    depthd = np.float32(depthd / 1000)
    T_camd_in_caml = caml.get_T_cam2_in_self(camd)
    with boxx.timeit("project_cam2_depth"):
        depthl = caml.project_cam2_depth(camd, depthd, T_camd_in_caml)
        caml.vis_project_align(imgl, depthl)
    stereo = Stereo(caml, camr)

    stereo2 = Stereo.load(stereo.dump(return_dict=1))
    T = camr.get_T_cam2_in_self(caml)
    stereo2.T, stereo2.t = T[:3, :3], T[:3, 3:]
    stereo2._get_undistort_rectify_map()

    stereo3 = Stereo(caml, camr, force_same_intrinsic=True)

    if "vis_stereo" and 1:
        visn = 1
        Cam.vis_stereo(caml, camr, stereo, visn)
        Cam.vis_stereo(caml, camr, stereo2, visn)
        Cam.vis_stereo(caml, camr, stereo3, visn)
        print(
            "|stereo2.t - stereo.t|: %.2fmm"
            % (np.sum((stereo2.t - stereo.t) ** 2) ** 0.5 * 1000)
        )
        print(
            "|stereo3.t - stereo.t|: %.2fmm"
            % (np.sum((stereo3.t - stereo.t) ** 2) ** 0.5 * 1000)
        )

    stereo_matching = SemiGlobalBlockMatching({})
    [
        s.set_stereo_matching(
            stereo_matching, max_depth=3, translation_rectify_img=True
        )
        for s in [stereo, stereo2, stereo3,]
    ]

    depths = [
        s.get_depth(imgl, imgr)["unrectify_depth"] for s in [stereo, stereo2, stereo3,]
    ]

    boxx.shows([undistort_imgl, list(map(vis_depth, [depthl] + depths))])

    se = boxx.sliceInt[500:1000, 1600:2200]
    sds = list(map(lambda d: d[se], [depthl] + depths))
    boxx.show(undistort_imgl[se], list(map(vis_depth, sds)))

    print_value = lambda d: (1000 * d[se][y : y + 10, x : x + 10].mean()).round(2)
    y, x = 0, 0
    print("mm", list(map(print_value, depths)))
    y, x = -20, 350
    print("mm", list(map(print_value, depths)))
