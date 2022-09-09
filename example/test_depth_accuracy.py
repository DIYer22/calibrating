#!/usr/bin/env python3

import cv2
import boxx
from boxx import imread, np

with boxx.impt(".."):
    from calibrating import *
    from calibrating import (
        Cam,
        Stereo,
        SemiGlobalBlockMatching,
        vis_depth,
        get_test_cams,
        utils,
    )
    import calibrating


class StereoDifferentK(calibrating.Stereo):
    pass
    # different K

    def _get_undistort_rectify_map(self):
        self.big_fov_cam = (
            self.cam1 if self.cam1.fovs["fov"] > self.cam2.fovs["fov"] else self.cam2
        )

        xy = self.big_fov_cam.xy
        self.R1, self.R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
            self.cam1.K, self.cam1.D, self.cam2.K, self.cam2.D, xy, self.R, self.t
        )

        self.undistort_rectify_map1 = cv2.initUndistortRectifyMap(
            self.cam1.K, self.cam1.D, self.R1, self.big_fov_cam.K, xy, cv2.CV_32FC1
        )
        self.undistort_rectify_map2 = cv2.initUndistortRectifyMap(
            self.cam2.K, self.cam2.D, self.R2, self.big_fov_cam.K, xy, cv2.CV_32FC1
        )

    def unrectify_depth(self, depth):
        T = np.eye(4)
        T[:3, :3] = self.R1.T
        depth_in_cam1 = Cam.project_cam2_depth(self.cam1, self.big_fov_cam, depth, T)
        return depth_in_cam1


if __name__ == "__main__":
    from boxx import *

    feature_type = "checkboard"
    feature_type = "aruco"
    is_dense = True
    vis_depth1 = lambda d: vis_depth(d, fix_range=(0, 3.5), slicen=30)

    cams = get_test_cams(feature_type)
    caml, camr, camd = cams["caml"], cams["camr"], cams["camd"]
    feature_lib = caml.feature_lib
    # caml, camr = camd, camr

    key = caml.valid_keys_intersection(camd)[0]
    imgl = imread(caml[key]["path"])
    imgr = imread(camr[key]["path"])
    imgd = imread(camd[key]["path"])
    depthd = imread(camd[key]["path"].replace("color.jpg", "depth.png")) / 1000
    # depthd = np.float32(depthd / 1000)
    undistort_imgl = cv2.undistort(imgl, caml.K, caml.D)
    stereo = Stereo(caml, camr)
    # Cam.vis_stereo(caml, camr, stereo)

    # get depth_board by chboard T
    depth_board = caml.get_calibration_board_depth(caml[key]["path"])["depth"]

    # get depthl by depthd
    T_camd_in_caml = caml.get_T_cam2_in_self(camd)
    with boxx.timeit("project_cam2_depth"):
        depthl = caml.project_cam2_depth(camd, depthd, T_camd_in_caml)

    stereo2 = Stereo.load(stereo.dump(return_dict=1))
    T = camr.get_T_cam2_in_self(caml)
    stereo2.T, stereo2.t = T[:3, :3], T[:3, 3:]
    # stereo2._get_undistort_rectify_map()

    # get depth_fm by MatchingByFeatureLib
    feature_matching = calibrating.MatchingByFeatureLib(feature_lib)
    stereo.set_stereo_matching(
        feature_matching, max_depth=3, translation_rectify_img=True
    )
    re_fm = stereo.get_depth(imgl, imgr)
    depth_fm = re_fm["unrectify_depth"]

    # get depth_sgbm by sgbm
    sgbm = SemiGlobalBlockMatching({})
    stereo.set_stereo_matching(sgbm, max_depth=3, translation_rectify_img=True)
    re_sgbm = stereo.get_depth(imgl, imgr)
    depth_sgbm = re_sgbm["unrectify_depth"]

    depths = [depthl, depth_board, depth_fm, depth_sgbm]
    boxx.shows([undistort_imgl, [vis_depth1(d) for d in depths]])

    se = boxx.sliceInt[500:1000, 1600:2200]
    sds = list(map(lambda d: d[se], depths))
    boxx.show(undistort_imgl[se], list(map(vis_depth1, sds)))
    print_value = lambda d: (1000 * d[se][y : y + 10, x : x + 10].mean()).round(2)
    y, x = 0, 0
    print("mm", list(map(print_value, depths)))
    y, x = -20, 350
    print("mm", list(map(print_value, depths)))
