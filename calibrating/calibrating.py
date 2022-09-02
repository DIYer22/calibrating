#!/usr/bin/env python3

import boxx
from boxx import imread

import os
import cv2
import yaml
import copy
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm

with boxx.inpkg():
    from .feature_libs import CheckboardFeatureLib
    from .stereo import Stereo
    from .utils import r_t_to_T, intrinsic_format_conversion
    from . import utils
    from .__info__ import __version__

TEMP = __import__("tempfile").gettempdir()


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
        self.img_paths = img_paths
        self.feature_lib = feature_lib
        self.save_feature_vis = save_feature_vis
        self.name = name or ("cam" + str(boxx.increase("caibrating.cam-name")))
        self.undistorted = undistorted
        self.enable_cache = enable_cache

        if img_paths is None:
            return
        # papre name to dict for calibrate
        assert len(img_paths), img_paths
        assert len(img_paths) == len(set(img_paths))
        if isinstance(img_paths, (list, tuple)):
            left_idx, right_idx = self.get_key_idx(img_paths)
            img_paths = {path[left_idx:right_idx]: path for path in sorted(img_paths)}
        self.img_paths = img_paths
        if self._cache():
            return
        for key in tqdm(sorted(img_paths)):
            path = img_paths[key]
            d = dict(path=path, img=self.process_img(imread(path)))
            feature_lib.find_image_points(d)
            d.pop("img")
            self[key] = d
        self.calibrate()

    def calibrate(self):
        self.set_xy()
        assert len(self.valid_keys), "No any valid image!"
        self.object_points = self._get_points_for_cv2("object_points")
        self.image_points = self._get_points_for_cv2("image_points")
        flags = 0
        if self.undistorted:
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
        init_K, init_D = None, None
        # flags=cv2.CALIB_FIX_ASPECT_RATIO#+cv2.CALIB_FIX_PRINCIPAL_POINT
        self.retval, self.K, self.D, rvecs, tvecs = cv2.calibrateCamera(
            self.object_points, self.image_points, self.xy, init_K, init_D, flags=flags,
        )

        for idx, key in enumerate(self.valid_keys):
            d = self[key]
            d["T"] = r_t_to_T(rvecs[idx], tvecs[idx])

        self._cache(do_cache=self.enable_cache)
        if self.save_feature_vis:
            visdir = TEMP + "/calibrating-vis-" + self.name
            print("\nSave visualization of feature points in dir:", visdir)
            os.makedirs(visdir, exist_ok=True)
            for key in tqdm(self.valid_keys):
                d = self[key]
                d["img"] = self.process_img(imread(d["path"]))
                vis = d.get("feature_lib", self.feature_lib).vis(d, self)
                boxx.imsave(visdir + "/" + key + ".jpg", vis)
                d.pop("img")

    @property
    def valid_keys(self):
        return set([key for key in self if len(self[key].get("image_points", {}))])

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
            try:
                pickle.dump(self, open(cache_path, "wb"))
            except TypeError:
                feature_lib = self.__dict__.pop("feature_lib")
                pickle.dump(self, open(cache_path, "wb"))
                self.feature_lib = feature_lib
            print(self)
        return False

    def process_img(self, img):
        """
        rotate90 img if xy != self.xy
        """
        xy = img.shape[1], img.shape[0]
        self.set_xy(img)
        if xy == self.xy:
            pass
        elif xy[::-1] == self.xy:
            img = np.ascontiguousarray(np.rot90(img))
        else:
            assert set(xy) == set(self.xy), f"{xy} != {self.xy}"
        return img

    def set_xy(self, img=None):
        if not hasattr(self, "xy"):
            if img is None:
                img = imread(next(iter(self.values()))["path"])
            self.xy = img.shape[1], img.shape[0]
        return self.xy

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

    def get_calibration_board_T(self, img, feature_lib=None):
        if feature_lib is None:
            assert hasattr(self, "feature_lib"), "Please set feature_lib"
            feature_lib = self.feature_lib
        assert img.shape[:2][::-1] == self.xy
        d = dict(img=img)
        feature_lib.find_image_points(d)
        if "image_points" in d:
            if isinstance(d["image_points"], dict):
                d["image_points"] = np.concatenate(
                    [d["image_points"][k] for k in sorted(d["image_points"])], 0
                )
                d["object_points"] = np.concatenate(
                    [d["object_points"][k] for k in sorted(d["object_points"])], 0
                )
            retval, rvecs, tvecs, reprojectionError = cv2.solvePnPGeneric(
                d["object_points"], d["image_points"][:, None], self.K, self.D
            )
            T_board_in_cam = utils.r_t_to_T(rvecs[0], tvecs[0])
            d.update(
                T=T_board_in_cam, reprojectionError=reprojectionError, retval=retval
            )
        return d

    def project_points(self, xyzs, T=None):
        if T is None:
            rvec, tvec = np.zeros((3, 1)), np.zeros((3, 1))
        else:
            rvec, tvec = utils.T_to_r_t(T)
        return cv2.projectPoints(xyzs, rvec, tvec, self.K, self.D)[0][:, 0]

    def undistort_points(self, uvs):
        K = self.K
        if uvs.ndim == 2:
            uvs = uvs[:, None]
        normalized_undistort_points = cv2.undistortPoints(uvs, K, self.D)[:, 0]
        return normalized_undistort_points * [[K[0, 0], K[1, 1]]] + [K[:2, 2]]

    def get_T_cam2_in_self(cam1, cam2):
        Ts = []
        keys = cam1.valid_keys_intersection(cam2)
        for key in keys:
            Ts.append(cam1[key]["T"] @ np.linalg.inv(cam2[key]["T"]))

        T = utils.mean_Ts(Ts)
        return T

    def project_cam2_depth(cam1, cam2, depth2, T=None, interpolation=1.5):
        if T is None:
            T = cam1.get_T_cam2_in_self(cam2)
        if interpolation:
            interpolation_rate = cam1.K[0, 0] / cam2.K[0, 0] * interpolation
            if interpolation >= 1:
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

    def vis_image_points_cover(self):
        # all_image_points = np.concatenate(self._get_points_for_cv2())
        # vis = utils.vis_point_uvs(all_image_points, self.xy[::-1])
        vis = self.xy[::-1]
        for uvs in self._get_points_for_cv2():
            vis = utils.vis_point_uvs(uvs, vis, convex_hull=True)

        all_object_points = np.concatenate(self._get_points_for_cv2("object_points"))
        object_points = np.unique(all_object_points, axis=0)
        t_middle = np.mean(object_points, 0)
        size_y = sorted(object_points.max(0) - object_points.min(0))[1]
        for d in self.values():
            if "T" in d:
                T = d["T"] @ r_t_to_T(np.zeros((3,)), t_middle)
                vis = utils.vis_T(T, self, vis, length=size_y / 2)
        return vis

    def _get_points_for_cv2(self, name="image_points"):
        points = []
        for key in self.valid_keys:
            point = self[key][name]
            if isinstance(point, dict):
                point = np.concatenate([point[id] for id in sorted(point)], axis=0)
            points.append(point)
        return points

    def valid_keys_intersection(cam1, cam2):
        return sorted(cam1.valid_keys.intersection(cam2.valid_keys))

    def __str__(self,):
        def with_delta(v, base):
            sub = v - base
            s = f"{round(v, 1)} Δ{'' if sub<0 else '+'}{round(sub, 1)}({round(abs(sub)/base*100, 1)}%)"
            return s

        s = (
            "Cam(name='%s'): \n\txy: %s \n\tfovs: %s \n\tK: %s\n\tD: %s \n\tvalid=%s/%s \n\tretval=%f \n\timage_path=%s \n"
            % (
                self.name,
                self.xy,
                utils._str_angle_dic(self.fovs),
                "\n\t\tfx=%s, fy=%s"
                % (round(self.fx, 1), with_delta(self.fy, self.fx),)
                + "\n\t\tcx=%s, cy=%s"
                % (
                    with_delta(self.cx, self.xy[0] / 2),
                    with_delta(self.cy, self.xy[1] / 2),
                ),
                str(self.D.round(2)),
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
            if k in ["D", "xy", "name", "T_in_main_cam", "retval"]
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
            dic = copy.deepcopy(path_or_str_or_dict)
        dic["K"] = intrinsic_format_conversion(dic)
        [dic.pop(k) for k in ("fx", "fy", "cx", "cy")]
        dic["D"] = np.float64(dic["D"])
        dic["xy"] = tuple(dic["xy"])
        self.__dict__.update(dic)
        return self

    def copy(self):
        new = type(self)()
        new.load(self.dump())
        return new

    def rotate(self, k=1):
        # TODO rotate consider cam.D
        # assert cam.dump()==cam.rotate(4).dump()
        dic = self.dump(return_dict=True)
        for idx in range(k):
            dic["xy"] = dic["xy"][::-1]
            dic["fx"], dic["fy"] = dic["fy"], dic["fx"]
            dic["cx"], dic["cy"] = dic["xy"][0] - dic["cy"], dic["cx"]
        new = type(self).load(dic)
        return new

    @property
    def fx(self):
        return self.K[0, 0]

    @property
    def fy(self):
        return self.K[1, 1]

    @property
    def cx(self):
        return self.K[0, 2]

    @property
    def cy(self):
        return self.K[1, 2]

    @property
    def fovs(self):
        fovx = 2 * boxx.arctan(self.xy[0] / 2 / self.fx)
        fovy = 2 * boxx.arctan(self.xy[1] / 2 / self.fy)
        fov = 2 * boxx.arctan(
            (boxx.tan(fovx / 2) ** 2 + boxx.tan(fovy / 2) ** 2) ** 0.5
        )
        return dict(fov=fov, fovx=fovx, fovy=fovy)

    @staticmethod
    def get_key_idx(paths):
        if len(paths) <= 1:
            return -3, None
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

    @classmethod
    def get_example_720p(cls):
        return cls.load(
            dict(
                fx=1000,
                fy=1000,
                cx=640,
                cy=360,
                D=np.zeros((1, 5)),
                xy=(1280, 720),
                name="example_720p",
            )
        )

    @classmethod
    def init_by_K_D(cls, K, D, xy, name=None):
        self = cls()
        self.name = name or ("cam" + str(boxx.increase("caibrating.cam-name")))
        self.K = K
        self.D = D
        self.xy = xy
        return self

    @classmethod
    def build_with_multi_feature_libs(cls, imgps, feature_libs, **kwargs):
        self = cls(**kwargs)
        if not isinstance(feature_libs, dict):
            feature_libs = dict([(str(i), v) for i, v in enumerate(feature_libs)])
        kwargs["save_feature_vis"] = False
        for name in feature_libs:
            kwargs["name"] = self.name + "~cam_~" + name
            feature_lib = feature_libs[name]
            cam_ = cls(imgps, feature_lib=feature_lib, **kwargs)
            for k in cam_:
                cam_[k]["feature_lib"] = feature_lib
                self[k + "~" + name] = cam_[k]
        self.calibrate()
        return self


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
    print(stereo)

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
