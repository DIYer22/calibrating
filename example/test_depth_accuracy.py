#!/usr/bin/env python3

import cv2
import boxx
from boxx import os, glob, imread, np
from calibrating import *
from calibrating import Cam, SemiGlobalBlockMatching, vis_depth, get_test_cams, utils
import calibrating


class Stereo(calibrating.Stereo):
    pass

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


class FeatureMatching(calibrating.MetaStereoMatching):
    def __init__(self, feature_lib, dense_predict=False):
        self.feature_lib = feature_lib
        self.dense_predict = dense_predict

    def __call__(self, img1, img2):
        self.feature_lib
        d1 = dict(img=img1)
        self.feature_lib.find_image_points(d1)
        image_points1 = d1["image_points"]
        d2 = dict(img=img2)
        self.feature_lib.find_image_points(d2)
        image_points2 = d2["image_points"]

        rectify_std = np.std((image_points2 - image_points1)[:, 1])
        point_disps = (image_points1 - image_points2)[:, 0]
        xyds = np.append(image_points1, point_disps[:, None], axis=-1)
        disp = utils.xyzs_to_arr2d(xyds, img1.shape[:2])
        return disp


def sparse_to_points(sparse):
    y, x = sparse.shape
    ys, xs = np.mgrid[:y, :x]
    mask = sparse != 0
    points = np.array([xs[mask], ys[mask], sparse[mask]]).T
    return points


def fit(xyzs, fit_xys):
    import scipy.interpolate

    spline = scipy.interpolate.Rbf(
        xyzs[:, 0],
        xyzs[:, 1],
        xyzs[:, 2],
        function="thin_plate",
        smooth=0.5,
        episilon=5,
    )
    zs = spline(fit_xys[:, 0], fit_xys[:, 1],)
    fit_xyzs = np.append(fit_xys, zs[:, None], axis=-1)
    return fit_xyzs


def dense_2d(sparse, constrained_type=None):
    points = sparse_to_points(sparse)

    mask = np.zeros_like(sparse, np.uint8)
    if constrained_type is not None:
        convex_hull = cv2.convexHull(np.int32(points[:, :2].round()))
        cv2.drawContours(mask, [convex_hull], -1, 255, -1)
        fit_xys = sparse_to_points(mask)[:, :2]
    else:
        fit_xys = sparse_to_points(~mask)
    fit_xyzs = fit(points, fit_xys)
    dense = utils.xyzs_to_arr2d(fit_xyzs, sparse.shape)
    return dense


if __name__ == "__main__":
    from boxx import *

    feature_type = "checkboard"
    feature_type = "aruco"
    is_dense = True

    cams = get_test_cams(feature_type)
    caml, camr, camd = cams["caml"], cams["camr"], cams["camd"]
    feature_lib = caml.feature_lib
    caml, camr = camd, camr

    key = caml.valid_keys_intersection(camd)[0]
    imgl = imread(caml[key]["path"])
    imgr = imread(camr[key]["path"])
    color_path_d = camd[key]["path"]
    depthd = imread(color_path_d.replace("color.", "depth.").replace(".jpg", ".png"))
    undistort_imgl = cv2.undistort(imgl, caml.K, caml.D)
    stereo = Stereo(caml, camr)
    Cam.vis_stereo(caml, camr, stereo)

    # get depth_board by chboard T
    T_board = caml[key]["T"]
    xyz_in_caml = utils.apply_T_to_point_cloud(T_board, feature_lib.object_points)
    depth_board = sparse_board = utils.point_cloud_to_depth(
        xyz_in_caml, caml.K, caml.xy
    )
    if is_dense:
        depth_board = dense_2d(sparse_board, "convex_hull")

    # get depthl by depthd
    depthd = np.float32(depthd / 1000)
    T_camd_in_caml = caml.get_T_cam2_in_self(camd)
    with boxx.timeit("project_cam2_depth"):
        depthl = caml.project_cam2_depth(camd, depthd, T_camd_in_caml)

    stereo2 = Stereo.load(stereo.dump(return_dict=1))
    T = camr.get_T_cam2_in_self(caml)
    stereo2.T, stereo2.t = T[:3, :3], T[:3, 3:]
    stereo2._get_undistort_rectify_map()

    # get depth_fm by FeatureMatching
    feature_matching = FeatureMatching(feature_lib, False)
    stereo.set_stereo_matching(
        feature_matching, max_depth=3, translation_rectify_img=True
    )
    depth_fm = sparse_fm = stereo.get_depth(imgl, imgr)["unrectify_depth"]
    if is_dense:
        depth_fm = dense_2d(sparse_fm, "convex_hull")

    # get depth_sgbm by sgbm
    sgbm = SemiGlobalBlockMatching({})
    stereo.set_stereo_matching(sgbm, max_depth=3, translation_rectify_img=True)
    depth_sgbm = stereo.get_depth(imgl, imgr)["unrectify_depth"]

    depths = [depthl, depth_board, depth_fm, depth_sgbm]
    boxx.shows([undistort_imgl, list(map(vis_depth, depths))])

    se = boxx.sliceInt[500:1000, 1600:2200]
    sds = list(map(lambda d: d[se], depths))
    boxx.show(undistort_imgl[se], list(map(vis_depth, sds)))
    print_value = lambda d: (1000 * d[se][y : y + 10, x : x + 10].mean()).round(2)
    y, x = 0, 0
    print("mm", list(map(print_value, depths)))
    y, x = -20, 350
    print("mm", list(map(print_value, depths)))
