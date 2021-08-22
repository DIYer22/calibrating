#!/usr/bin/env python3

import boxx
from boxx import npa, imread

import os
import cv2
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm

with boxx.inpkg():
    from .stereo import Stereo
    from .utils import r_t_to_T
    from . import utils


TEMP = __import__("tempfile").gettempdir()


class MetaFeatureLib:
    def find_image_points(self, d):
        raise NotImplementedError()

    @staticmethod
    def object_points(self):
        raise NotImplementedError()

    def vis(self, d, cam=None):
        raise NotImplementedError()

    def set_Ts(self, d, cam=None):
        if "T" in d:
            d["Ts"] = [d["T"]]
        return d.get("Ts", [])


class CheckboardFeatureLib(MetaFeatureLib):
    def __init__(self, checkboard=(11, 8), gap_mm=25):
        self.checkboard = checkboard
        self.gap_mm = gap_mm
        self.object_points = np.zeros(
            (self.checkboard[0] * self.checkboard[1], 3), np.float32
        )
        self.object_points[:, :2] = np.float32(
            np.mgrid[: self.checkboard[0], : self.checkboard[1]].T.reshape(-1, 2)
            * self.gap_mm
            / 1000
        )

    def find_image_points(self, d):
        img = d["img"]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.checkboard, None)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        if ret:
            cv2.cornerSubPix(
                gray, corners, (4, 4), (-1, -1), criteria,
            )
            d["corners"] = corners
            d["image_points"] = corners

    def vis(self, d, cam=None):
        img = boxx.imread(d["path"])
        h, w = img.shape[:2]
        draw_subpix = h < 1500
        if draw_subpix:
            sub_size = 2
            img = cv2.resize(img, (w * sub_size, h * sub_size))
            cv2.drawChessboardCorners(
                img, self.checkboard, d["corners"] * sub_size, True
            )
        else:
            cv2.drawChessboardCorners(img, self.checkboard, d["corners"], True)
        return img


class ArucoFeatureLib(MetaFeatureLib):
    def __init__(self):
        import cv2.aruco

        aruco_temp_str = "480 240 0 580 240 0 580 340 0 480 340 0 480 120 0 580 120 0 580 220 0 480 220 0 480 0 0 580 0 0 580 100 0 480 100 0 0 360 0 100 360 0 100 460 0 0 460 0 480 360 0 580 360 0 580 460 0 480 460 0 480 480 0 580 480 0 580 580 0 480 580 0 360 480 0 460 480 0 460 580 0 360 580 0 240 480 0 340 480 0 340 580 0 240 580 0 120 480 0 220 480 0 220 580 0 120 580 0 0 480 0 100 480 0 100 580 0 0 580 0 0 240 0 100 240 0 100 340 0 0 340 0 0 120 0 100 120 0 100 220 0 0 220 0 0 0 0 100 0 0 100 100 0 0 100 0 120 0 0 220 0 0 220 100 0 120 100 0 240 0 0 340 0 0 340 100 0 240 100 0 360 0 0 460 0 0 460 100 0 360 100 0 120 120 0 220 120 0 220 220 0 120 220 0 240 120 0 340 120 0 340 220 0 240 220 0 360 120 0 460 120 0 460 220 0 360 220 0 120 360 0 220 360 0 220 460 0 120 460 0 360 360 0 460 360 0 460 460 0 360 460 0 240 360 0 340 360 0 340 460 0 240 460 0 120 240 0 220 240 0 220 340 0 120 340 0 360 240 0 460 240 0 460 340 0 360 340 0"
        self.object_points = np.float32(
            npa(boxx.findints(aruco_temp_str)).reshape(-1, 3) / 1000.0
        )
        self.aruco_dict_idx = cv2.aruco.DICT_6X6_250

    def find_image_points(self, d):
        img = d["img"]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        parameters = cv2.aruco.DetectorParameters_create()
        d["corners"], d["ids"], rejectedImgPoints = cv2.aruco.detectMarkers(
            gray, cv2.aruco.Dictionary_get(self.aruco_dict_idx), parameters=parameters
        )
        d["valid"] = d["ids"] is not None and len(d["ids"]) * 4 == len(
            self.object_points
        )
        if d["valid"]:
            d["sorted_corners"] = np.concatenate(
                [
                    point
                    for idx, point in sorted(
                        enumerate(d["corners"]), key=lambda x: d["ids"][x[0]]
                    )
                ]
            )
            d["image_points"] = d["sorted_corners"].reshape(-1, 2)[:None]

    def vis(self, d, cam=None):
        img = boxx.imread(d["path"])
        cv2.aruco.drawDetectedMarkers(img, d["corners"], d["ids"])
        if cam is not None and len(d["corners"]):
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                d["corners"], 0.1, cam.K, cam.D
            )
            for i in range(rvec.shape[0]):
                cv2.aruco.drawAxis(img, cam.K, cam.D, rvec[i], tvec[i], 0.05)
        return img

    def set_Ts_by_markers(self, d, cam=None):
        if "image_points" in d:
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                d["sorted_corners"], 0.1, cam.K, cam.D
            )

            d["Ts"] = [utils.r_t_to_T(r, t) for r, t in zip(rvec, tvec)]

    # set_Ts = set_Ts_by_markers


class Cam(dict):
    def __init__(
        self,
        img_paths,
        feature_lib=None,
        save_feature_vis=True,
        name=None,
        enable_cache=False,
    ):
        assert len(img_paths) and len(img_paths) == len(set(img_paths))
        super().__init__()
        self.feature_lib = feature_lib
        if isinstance(img_paths, (list, tuple)):
            left_idx, right_idx = self.get_key_idx(img_paths)
            img_paths = {path[left_idx:right_idx]: path for path in sorted(img_paths)}

        self.img_paths = img_paths
        self.name = name or ("cam" + str(boxx.increase("caibrating.cam-name")))

        self.enable_cache = enable_cache
        if self._cache():
            return
        for key in tqdm(sorted(img_paths)):
            path = img_paths[key]
            d = {}
            d["path"] = path

            d["img"] = boxx.imread(path)
            self.xy = d["img"].shape[1], d["img"].shape[0]
            feature_lib.find_image_points(d)
            d.pop("img")
            self[key] = d
            # break

        valid_keys = self.valid_keys()
        self.image_points = [self[key]["image_points"] for key in valid_keys]
        self.object_points = feature_lib.object_points
        boxx.g()
        self.retval, self.K, self.D, rvecs, tvecs = cv2.calibrateCamera(
            [feature_lib.object_points] * len(self.image_points),
            self.image_points,
            self.xy,
            None,
            None,
        )

        for idx, key in enumerate(valid_keys):
            d = self[key]
            d["T"] = r_t_to_T(rvecs[idx], tvecs[idx])
            feature_lib.set_Ts(d, self)

        if save_feature_vis:
            visdir = TEMP + "/calibrating-vis-" + self.name
            os.makedirs(visdir, exist_ok=True)
            for key, d in tqdm(self.items()):
                vis = feature_lib.vis(d, self)
                boxx.imsave(visdir + "/" + key + ".jpg", vis)
            print("Save image_points vis in dir:", visdir)
        self._cache(do_cache=True)

    def _cache(self, do_cache=False):
        cache_path = TEMP + "/calibrating-cache-" + self.name + ".pkl"
        if self.enable_cache and os.path.isfile(cache_path):
            loaded = pickle.load(open(cache_path, "rb"))
            if set(self.img_paths.values()) == set(loaded.img_paths.values()):
                self.clear()
                self.update(loaded)
                self.__dict__.clear()
                self.__dict__.update(loaded.__dict__)
                print("Load from cache:", cache_path)
                print(self)
                return True
        if do_cache:
            pickle.dump(self, open(cache_path, "wb"))
            print(self)
        return False

    def stereo(caml, camr):
        K1, D1, K2, D2 = (
            caml.K,
            caml.D,
            camr.K,
            camr.D,
        )
        xy = caml.xy

        keys = caml.valid_keys_intersection(camr)
        l1 = [caml[key]["image_points"] for key in keys]
        l2 = [camr[key]["image_points"] for key in keys]
        object_points = [caml.object_points] * len(keys)
        return Stereo.init_by_calibrate(K1, D1, K2, D2, xy, object_points, l1, l2)

    def vis_stereo(caml, camr, stereo=None, visn=4):
        if stereo is None:
            stereo = caml.align_stereo(camr)
        visdir = TEMP + "/calibrating-stereo-vis/"
        os.makedirs(visdir, exist_ok=True)
        for key in caml.valid_keys_intersection(camr)[:visn]:
            imgl = boxx.imread(caml[key]["path"])
            imgr = boxx.imread(camr[key]["path"])
            vis_align = stereo.vis(stereo.align([imgl, imgr]))
            boxx.imsave(visdir + key + ".jpg", vis_align)
        print("Save stereo vis to:", visdir)
        stereo.shows(stereo.align([imgl, imgr]))
        return stereo

    def get_T_cam2_in_self(cam1, cam2):
        Ts = []
        keys = cam1.valid_keys_intersection(cam2)
        for key in keys:
            Ts.extend(
                [
                    Tl @ np.linalg.inv(Td)
                    for Tl, Td in zip(cam1[key]["Ts"], cam2[key]["Ts"])
                ]
            )

        T = utils.mean_Ts(Ts)
        return T

    def project_cam2_depth(cam1, cam2, depth2, T=None):
        if T is None:
            T = cam1.get_T_cam2_in_self(cam2)
        point_cloud2 = utils.depth_to_point_cloud(depth2, cam2.K)
        point_cloud1 = utils.apply_T_to_point_cloud(T, point_cloud2)
        depth1 = utils.point_cloud_to_depth(point_cloud1, cam1.K, cam1.xy)
        return depth1

    def vis_project_align(self, img, depth, undistort=False):
        if undistort:
            img = cv2.undistort(img, self.K, self.D)
        depth_uint8 = np.uint8(depth / depth.max() * 255)
        depth_vis = np.uint8(cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET) * 0.75)
        n_line = 21
        y, x = img.shape[:2]
        # TODO better vis code
        visv = Stereo.vis([img, depth_vis], n_line=n_line)
        vis = np.concatenate((visv, Stereo.vis([depth_vis, img], n_line=n_line),), 0)
        vis = np.rot90(
            Stereo.vis(
                [np.rot90(vis[:y]), np.rot90(vis[y:])], n_line=int(n_line * x * 2 / y)
            ),
            3,
        )
        viss = [vis[:y, :x], vis[:y, x:], vis[y:, :x], vis[y:, x:]]
        boxx.shows(viss)
        return viss

    def valid_keys(self):
        return set([key for key in self if "image_points" in self[key]])

    def valid_keys_intersection(cam1, cam2):
        return sorted(cam1.valid_keys().intersection(cam2.valid_keys()))

    def __str__(self,):
        s = "cam-%s: \n\tvalid=%s/%s \n\tretval=%.2f \n\timage_path=%s\n" % (
            self.name,
            len(self.image_points),
            len(self),
            self.retval,
            next(iter(self.values()))["path"],
        )
        return s

    __repr__ = __str__

    @staticmethod
    def get_key_idx(paths):
        if len(paths) <= 1:
            return 0, 1
        left_idx = 0
        while all([paths[0][left_idx] == path[left_idx] for path in paths]):
            left_idx += 1
        right_idx = -1
        while all([paths[0][right_idx] == path[right_idx] for path in paths]):
            right_idx -= 1
        return left_idx, right_idx + 1


if __name__ == "__main__":
    from boxx import *

    root = os.path.abspath(
        os.path.join(
            __file__, "../../../calibrating_example_data/paired_stereo_and_depth_cams"
        )
    )
    feature_lib = ArucoFeatureLib()
    caml = Cam(
        glob(os.path.join(root, "*", "stereo_l.jpg")),
        feature_lib,
        name="left",
        enable_cache=True,
    )
    camr = Cam(
        glob(os.path.join(root, "*", "stereo_r.jpg")),
        feature_lib,
        name="right",
        enable_cache=True,
    )
    camd = Cam(
        glob(os.path.join(root, "*", "depth_cam_color.jpg")),
        feature_lib,
        name="depth",
        enable_cache=True,
    )

    stereo = Cam.stereo(caml, camr)

    T = caml.get_T_cam2_in_self(camd)
    keys = caml.valid_keys_intersection(camd)
    key = keys[4]
    imgl = imread(caml[key]["path"])
    undis = cv2.undistort(imgl, caml.K, caml.D)
    color_path_d = camd[key]["path"]
    imgd = imread(color_path_d)
    depthd = imread(color_path_d.replace("color.jpg", "depth.png"))
    depthd = np.float32(depthd / 1000)

    depthl = caml.project_cam2_depth(camd, depthd, T)

    caml.vis_project_align(imgl, depthl, undistort=False)
