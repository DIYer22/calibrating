#!/usr/bin/env python3

import boxx
from boxx import imread

import os
import cv2
import yaml
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm

with boxx.inpkg():
    from .stereo import Stereo
    from .utils import r_t_to_T, intrinsic_format_conversion
    from . import utils
    from .__info__ import __version__


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
    def __init__(self, checkboard=(11, 8), size_mm=25):
        self.checkboard = checkboard
        self.size_mm = size_mm
        self.object_points = np.zeros(
            (self.checkboard[0] * self.checkboard[1], 3), np.float32
        )
        self.object_points[:, :2] = np.float32(
            np.mgrid[: self.checkboard[0], : self.checkboard[1]].T.reshape(-1, 2)
            * self.size_mm
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
            np.array(boxx.findints(aruco_temp_str)).reshape(-1, 3) / 1000.0
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
        img_paths=None,
        feature_lib=None,
        save_feature_vis=True,
        name=None,
        undistorted=False,
        enable_cache=False,
    ):
        super().__init__()
        if img_paths is None:
            return
        assert len(img_paths) and len(img_paths) == len(set(img_paths))
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
        flags = 0
        if undistorted:
            flags = cv2.CALIB_ZERO_TANGENT_DIST + sum(
                [
                    cv2.CALIB_FIX_K1,
                    cv2.CALIB_FIX_K2,
                    cv2.CALIB_FIX_K3,
                    cv2.CALIB_FIX_K4,
                    cv2.CALIB_FIX_K5,
                    cv2.CALIB_FIX_K6,
                ]
            )
        self.retval, self.K, self.D, rvecs, tvecs = cv2.calibrateCamera(
            [feature_lib.object_points] * len(self.image_points),
            self.image_points,
            self.xy,
            None,
            None,
            flags=flags,
        )

        for idx, key in enumerate(valid_keys):
            d = self[key]
            d["T"] = r_t_to_T(rvecs[idx], tvecs[idx])
            feature_lib.set_Ts(d, self)

        self._cache(do_cache=True)
        if save_feature_vis:
            visdir = TEMP + "/calibrating-vis-" + self.name
            print("\nSave visualization of feature points in dir:", visdir)
            os.makedirs(visdir, exist_ok=True)
            for key in tqdm(valid_keys):
                d = self[key]
                vis = feature_lib.vis(d, self)
                boxx.imsave(visdir + "/" + key + ".jpg", vis)

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

    def stereo_with(caml, camr):
        return Stereo(caml, camr)

    def vis_stereo(caml, camr, stereo=None, visn=1):
        if stereo is None:
            stereo = Stereo(caml, camr)
        visdir = TEMP + "/calibrating-stereo-vis/"
        print("Save stereo vis to:", visdir)
        os.makedirs(visdir, exist_ok=True)
        for key in tqdm(caml.valid_keys_intersection(camr)[:visn]):
            imgl = boxx.imread(caml[key]["path"])
            imgr = boxx.imread(camr[key]["path"])
            vis_rectify = stereo.vis(*stereo.rectify(imgl, imgr))
            boxx.imsave(visdir + key + ".jpg", vis_rectify)
        stereo.shows(*stereo.rectify(imgl, imgr))
        return stereo

    def get_T_cam2_in_self(cam1, cam2):
        Ts = []
        keys = cam1.valid_keys_intersection(cam2)
        for key in keys:
            Ts.extend(
                [
                    T1 @ np.linalg.inv(T2)
                    for T1, T2 in zip(cam1[key]["Ts"], cam2[key]["Ts"])
                ]
            )

        T = utils.mean_Ts(Ts)
        return T

    def project_cam2_depth(cam1, cam2, depth2, T=None, interpolation=1.5):
        if T is None:
            T = cam1.get_T_cam2_in_self(cam2)
        if interpolation:
            interpolation_rate = cam1.K[0, 0] / cam2.K[0, 0] * interpolation
            interpolation_rate = max(interpolation_rate, 1)
        else:
            interpolation_rate = 1
        point_cloud2 = utils.depth_to_point_cloud(
            depth2, cam2.K, interpolation_rate=interpolation_rate
        )
        point_cloud1 = utils.apply_T_to_point_cloud(T, point_cloud2)
        depth1 = utils.point_cloud_to_depth(point_cloud1, cam1.K, cam1.xy)
        return depth1

    def vis_project_align(self, img, depth):
        img = cv2.undistort(img, self.K, self.D)
        depth_uint8 = np.uint8(depth / depth.max() * 255)
        depth_vis = np.uint8(cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET) * 0.75)
        return utils.vis_align(img, depth_vis)

    def valid_keys(self):
        return set([key for key in self if "image_points" in self[key]])

    def valid_keys_intersection(cam1, cam2):
        return sorted(cam1.valid_keys().intersection(cam2.valid_keys()))

    def __str__(self,):
        s = (
            "Cam: \n\tname: '%s' \n\txy: %s \n\tfovs: %s \n\tvalid=%s/%s \n\tretval=%.2f \n\timage_path=%s \n"
            % (
                self.name,
                self.xy,
                utils._str_angle_dic(self.fovs),
                len(self.__dict__.get("image_points", [])),
                len(self),
                self.__dict__.get("retval", -1),
                len(self) and next(iter(self.values()))["path"],
            )
        )
        return s

    __repr__ = __str__

    def dump(self, path="", return_dict=False):

        dic = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in self.__dict__.items()
            if k in ["D", "xy", "name", "T_in_main_cam"]
        }
        dic.update(intrinsic_format_conversion(self.K))
        if return_dict:
            return dic
        dic["_calibrating_version"] = __version__
        yamlstr = yaml.safe_dump(dic)
        if path:
            with open(path, "w") as f:
                f.write(yamlstr)
        return yamlstr

    def load(self, path_or_str_or_dict=None):
        if path_or_str_or_dict is None:
            path_or_str_or_dict = self
            self = Cam()
        if not isinstance(path_or_str_or_dict, (list, dict)):
            path_or_str = path_or_str_or_dict
            if "\n" in path_or_str:
                dic = yaml.safe_load(path_or_str)
            else:
                with open(path_or_str) as f:
                    dic = yaml.safe_load(f)
        else:
            dic = path_or_str_or_dict
        dic["K"] = intrinsic_format_conversion(dic)
        dic["D"] = np.array(dic["D"])
        dic["xy"] = tuple(dic["xy"])
        self.__dict__.update(dic)
        return self

    def copy(self):
        new = type(self)()
        new.load(self.dump())
        return new

    @property
    def fx(self):
        return self.K[0, 0]

    @property
    def fy(self):
        return self.K[1, 1]

    @property
    def fovs(self):
        fovx = 2 * boxx.arctan(self.xy[0] / 2 / self.fx)
        fovy = 2 * boxx.arctan(self.xy[1] / 2 / self.fy)
        fov = 2 * boxx.arctan(
            (boxx.tan(fovx / 2) ** 2 + boxx.tan(fovy / 2) ** 2) ** 0.5
        )
        return dict(fov=fov, fovx=fovx, fovy=fovy)

    @classmethod
    def init_by_K_D(cls, K, D, xy, name=None):
        self = cls()
        self.name = name or ("cam" + str(boxx.increase("caibrating.cam-name")))
        self.K = K
        self.D = D
        self.xy = xy
        return self

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

    @staticmethod
    def get_test_cams():
        root = os.path.abspath(
            os.path.join(
                __file__,
                "../../../calibrating_example_data/paired_stereo_and_depth_cams_checkboard",
            )
        )
        feature_lib = CheckboardFeatureLib(checkboard=(7, 10), size_mm=22.564)
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
            undistorted=True,
            enable_cache=True,
        )
        return caml, camr, camd


class Cams(list):
    """
    Unified multi-camera format, including a main cam. And each cam has attr "T_in_main_cam"
    """

    def __init__(self, main_cam=None, *cams):
        super().__init__()
        if main_cam is None:
            return
        self.init_by_cams(main_cam, *cams)

    def init_by_cams(self, main_cam, *cams):
        self.append(main_cam)
        self.extend(cams)
        for cam in self:
            cam.T_in_main_cam = main_cam.get_T_cam2_in_self(cam)

    def dump(self, path=""):
        dic = [cam.dump(return_dict=True) for cam in self]
        yamlstr = yaml.safe_dump(dic)
        if path:
            with open(path, "w") as f:
                f.write(yamlstr)
        return yamlstr

    def load(self, path_or_str_or_list=None):
        if path_or_str_or_list is None:
            path_or_str_or_list = self
            self = Cams()
        if not isinstance(path_or_str_or_list, (list, dict)):
            path_or_str = path_or_str_or_list
            if "\n" in path_or_str:
                l = yaml.safe_load(path_or_str)
            else:
                with open(path_or_str) as f:
                    l = yaml.safe_load(f)
        else:
            l = path_or_str_or_list
        self.extend([Cam.load(dic) for dic in l])
        return self


if __name__ == "__main__":
    from boxx import *

    caml, camr, camd = Cam.get_test_cams()
    print(Cam.load(camd.dump()))

    stereo = Stereo(caml, camr)

    T_camd_in_caml = caml.get_T_cam2_in_self(camd)
    key = caml.valid_keys_intersection(camd)[0]
    imgl = imread(caml[key]["path"])
    color_path_d = camd[key]["path"]
    depthd = imread(color_path_d.replace("color.jpg", "depth.png"))
    depthd = np.float32(depthd / 1000)

    depthl = caml.project_cam2_depth(camd, depthd, T_camd_in_caml)
    # depthd_cycle = camd.project_cam2_depth(caml, depthl, camd.get_T_cam2_in_self(caml))
    # shows(depthd, depthd_cycle)

    caml.vis_project_align(imgl, depthl)
