#!/usr/bin/env python3

import boxx
from boxx import imread

import os
import cv2
import yaml
import copy
import uuid
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm

with boxx.inpkg():
    from . import utils
    from .stereo import Stereo
    from .__info__ import __version__
    from .utils import r_t_to_T, intrinsic_format_conversion
    from .reconstruction import convert_cam_to_nerf_json
    from .boards import Chessboard, BaseBoard

TEMP = __import__("tempfile").gettempdir()


class Cam(dict):
    def __init__(
        self,
        img_paths=None,
        board=None,
        save_feature_vis=True,
        name=None,
        undistorted=False,
        enable_cache=False,
    ):
        super().__init__()
        self.img_paths = img_paths
        self.board = board
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
            board.find_image_points(d)
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
            self.object_points,
            self.image_points,
            self.xy,
            init_K,
            init_D,
            flags=flags,
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
                vis = d.get("board", self.board).vis(d, self)
                boxx.imsave(visdir + "/" + key + ".jpg", vis)
                d.pop("img")

    def fine_tuning_intrinsic(self, base_cam, momenta=0.5):
        """
        Fine tuning intrinsic from base_cam's K and D.
        By add new image_points and object_points that fit base_cam's intrinsic
        Then run self.calibrate() again

        Parameters
        ----------
        base_cam : Cam
            base cam.
        momenta : float, [0~1]
            the momenta of base_cam's intrinsic. The default is .5.
        """
        if momenta in (0, 1):
            if momenta == 1:
                self.load(base_cam.dump())
            return self
        d0 = self[list(self)[0]]
        base_board = BaseBoard()
        points = self._get_points_for_cv2()
        newn = sum(map(len, points))
        needn = int(newn * momenta / (1 - momenta))
        uvs = np.random.rand(needn, 2) * [list(self.xy)]

        normalized_undistort_points = cv2.undistortPoints(
            uvs[:, None], base_cam.K, base_cam.D
        )[:, 0]
        xyzs = np.zeros_like(normalized_undistort_points, shape=(needn, 3))
        xyzs[:, :2] = normalized_undistort_points

        dn = int(len(points) * momenta / (1 - momenta))
        for d_idx in range(dn):
            d = dict(board=base_board)
            d["path"] = d0["path"]
            d["image_points"] = np.float32(uvs[d_idx::dn])
            d["object_points"] = np.float32(xyzs[d_idx::dn])
            self[f"~base_cam_points-{self.name}-{str(uuid.uuid1())}"] = d
        self.calibrate()
        return self

    @property
    def valid_keys(self):
        return set([key for key in self if len(self[key].get("image_points", {}))])

    def _cache(self, do_cache=False):
        cache_path = TEMP + "/calibrating-cache-" + self.name + ".pkl"
        if self.enable_cache and os.path.isfile(cache_path):
            try:
                loaded = pickle.load(open(cache_path, "rb"))
                if set(self.img_paths.values()) == set(loaded.img_paths.values()):
                    self.clear()
                    self.update(loaded)
                    self.__dict__.clear()
                    self.__dict__.update(loaded.__dict__)
                    print("Load from cache:", cache_path)
                    print(self)
                    return True
            except Exception as e:
                print(f"Load cache failed! Because {type(e).__name__}('{e})'")
        if do_cache:
            try:
                pickle.dump(self, open(cache_path, "wb"))
            except TypeError:
                board = self.__dict__.pop("board")
                pickle.dump(self, open(cache_path, "wb"))
                self.board = board
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

    def get_calibration_board_T(self, img_or_path_or_d, board=None):
        if isinstance(img_or_path_or_d, dict):
            d = img_or_path_or_d.copy()
            if "img" not in d:
                d["img"] = imread(d["path"])
        else:
            img = (
                imread(img_or_path_or_d)
                if isinstance(img_or_path_or_d, str)
                else img_or_path_or_d
            )
            d = dict(img=img)
        assert d["img"].shape[:2][::-1] == self.xy
        board = self.get_board(d, board)

        board.find_image_points(d)
        if "image_points" in d:
            d.update(self.perspective_n_point(d["image_points"], d["object_points"]))
        return d

    def get_calibration_board_depth(cam, img_or_path_or_d, is_dense=True, board=None):
        d = cam.get_calibration_board_T(img_or_path_or_d, board=board)
        if "object_points" not in d:
            d["depth"] = np.zeros(cam.xy[::-1])
            return d
        object_points = d["object_points"]
        if isinstance(object_points, dict):
            object_points = np.concatenate(list(d["object_points"].values()), 0)
        xyz_in_caml = utils.apply_T_to_point_cloud(d["T"], object_points)
        depth_board = sparse_board = utils.point_cloud_to_depth(
            xyz_in_caml, cam.K, cam.xy
        )
        if is_dense:
            depth_board = 1 / utils.interpolate_sparse2d(
                1 / sparse_board, "convex_hull"
            )
        d["depth"] = depth_board
        return d

    def get_board(cam=None, d=None, board=None):
        """
        get right board form cam, caboard d, or specified  board
        board > d["board"] > cam.board
        """
        if board is None:
            if d and d.get("board"):
                board = d.get("board")
            elif hasattr(cam, "board"):
                board = cam.board
        assert board, "Please set board or cam.board"
        return board

    def perspective_n_point(self, image_points, object_points):
        image_points = utils.convert_points_for_cv2(image_points)
        object_points = utils.convert_points_for_cv2(object_points)
        retval, rvecs, tvecs, reprojectionError = cv2.solvePnPGeneric(
            object_points, image_points[:, None], self.K, self.D
        )
        T = utils.r_t_to_T(rvecs[0], tvecs[0])
        return dict(T=T, retval=retval, reprojection_error=reprojectionError)

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
        interpolation_rate = utils._get_appropriate_interpolation_rate(
            cam1, cam2, interpolation
        )
        point_cloud2 = utils.depth_to_point_cloud(
            depth2, cam2.K, interpolation_rate=interpolation_rate
        )
        point_cloud1 = utils.apply_T_to_point_cloud(T, point_cloud2)
        depth1 = utils.point_cloud_to_depth(point_cloud1, cam1.K, cam1.xy)
        return depth1

    def vis_depth_alignment(self, img, depth):
        """
        Visualize alignment of img and depth using browser
        Note: img is raw img file without undistort
        """
        img = cv2.undistort(img, self.K, self.D)
        depth = depth.clip(0, 5000 if depth.dtype == np.uint16 else 5)
        depth_uint8 = np.uint8(depth / depth.max() * 255)
        depth_vis = np.uint8(cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET) * 0.75)
        return utils.vis_align(img, depth_vis)

    def vis_reproject_img_alignment(cam1, cam2, depth2, img2, img1, interpolation=1.5):
        """
        Visualize alignment of undistorted_img1 and reproject_img from img2
        Note: img is raw img file without undistort
        """
        undistorted_img1 = cv2.undistort(img1, cam1.K, cam1.D)
        assert not cam2.D.any(), f"cam2.D has distort: {cam2.D}"
        T_2in1 = cam1.get_T_cam2_in_self(cam2)
        interpolation_rate = utils._get_appropriate_interpolation_rate(
            cam1, cam2, interpolation
        )
        mapx, mapy = utils.get_reproject_remap(
            cam1.K,
            cam2.K,
            T_2in1,
            depth2,
            cam1.xy,
            interpolation_rate=interpolation_rate,
        )
        reproject_img = cv2.remap(img2, mapx, mapy, cv2.INTER_LINEAR)
        return utils.vis_align(undistorted_img1, reproject_img)

    def vis_image_points_cover(self):
        vis = utils._get_vis_background_of_cam(self)
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
            points.append(utils.convert_points_for_cv2(point))
        return points

    def valid_keys_intersection(cam1, cam2):
        return sorted(cam1.valid_keys.intersection(cam2.valid_keys))

    def __str__(self):
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
                % (
                    round(self.fx, 1),
                    with_delta(self.fy, self.fx),
                )
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
        dic["D"] = np.float64(dic["D"]) if "D" in dic else np.zeros((1, 5))
        dic["xy"] = tuple(dic.get("xy", getattr(self, "xy", "")))
        assert len(dic["xy"]), "Need xy"
        self.__dict__.update(dic)
        if len(self):
            self.update_intrinsic()
        return self

    def update_intrinsic(self):
        """
        When update intrinsic, will update T in all calibrate image dic by self.perspective_n_point()
        """
        for d in self.values():
            if "image_points" in d:
                d.update(
                    self.perspective_n_point(
                        d["image_points"],
                        d["object_points"],
                    )
                )

    convert_to_nerf_json = convert_cam_to_nerf_json

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
        board = Chessboard(checkboard=(7, 10), size_mm=22.564)
        caml = Cam(
            glob(os.path.join(root, "*", "stereo_l.jpg")),
            board,
            name="left",
            enable_cache=True,
        )
        camr = Cam(
            glob(os.path.join(root, "*", "stereo_r.jpg")),
            board,
            name="right",
            enable_cache=True,
        )
        camd = Cam(
            glob(os.path.join(root, "*", "depth_cam_color.jpg")),
            board,
            name="depth",
            enable_cache=True,
        )
        built_in_intrinsics = dict(
            fx=1474.1182177692722,
            fy=1474.125874583105,
            cx=1037.599716850734,
            cy=758.3072639103259,
        )
        # depth need to be used in pairs with camera's built-in intrinsics
        camd.load(built_in_intrinsics)
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
    # caml, camr, camd = utils.get_test_cams("aruco").values()
    print(Cam.load(camd.dump()))

    stereo = Stereo(caml, camr)
    print(stereo)

    T_camd_in_caml = caml.get_T_cam2_in_self(camd)
    key = caml.valid_keys_intersection(camd)[0]
    imgl = imread(caml[key]["path"])
    color_path_d = camd[key]["path"]
    imgd = imread(color_path_d)
    depthd = imread(color_path_d.replace("color.jpg", "depth.png"))
    depthd = np.float32(depthd / 1000)

    caml.vis_reproject_img_alignment(camd, depthd, imgd, imgl)

    depthl = caml.project_cam2_depth(camd, depthd, T_camd_in_caml)
    caml.vis_depth_alignment(imgl, depthl)
