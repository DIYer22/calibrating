#!/usr/bin/env python3
import cv2
import yaml
import boxx
import numpy as np

with boxx.inpkg():
    from . import utils
    from .__info__ import __version__


class Stereo:
    """
    Structure of Stereo
    ├── R: (3, 3)float64
    ├── t: (3, 1)float64
    ├── R1: (3, 3)float64
    ├── R2: (3, 3)float64
    ├── undistort_rectify_map1: tuple 2
    │   ├── 0: (3648, 5472)float32
    │   └── 1: (3648, 5472)float32
    ├── undistort_rectify_map2: tuple 2
    │   ├── 0: (3648, 5472)float32
    │   └── 1: (3648, 5472)float32
    ├── cam1: dict  2
    │   ├── K: (3, 3)float64
    │   └── D: (1, 5)float64
    └── cam2: dict  2
        ├── K: (3, 3)float64
        └── D: (1, 5)float64
    """

    def __init__(self, cam1=None, cam2=None, force_same_intrinsic=False):
        if cam1 is None:
            return
        self.cam1 = cam1
        self.cam2 = cam2
        self.force_same_intrinsic = force_same_intrinsic
        self.get_R_t_by_stereo_calibrate()
        self._get_undistort_rectify_map()

    def get_R_t_by_stereo_calibrate(self):
        keys = self.cam1.valid_keys_intersection(self.cam2)
        image_points1 = [self.cam1[key]["image_points"] for key in keys]
        image_points2 = [self.cam2[key]["image_points"] for key in keys]
        object_points = [self.cam1.object_points] * len(keys)
        if self.force_same_intrinsic:
            self.cam2 = self.cam2.copy()
            self.cam2.K = self.cam1.K
            self.cam2.D = self.cam1.D

        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC

        stereocalib_criteria = (
            cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
            100,
            1e-5,
        )
        ret, _, _, _, _, self.R, self.t, E, F = cv2.stereoCalibrate(
            object_points,
            image_points1,
            image_points2,
            self.cam1.K,
            self.cam1.D,
            self.cam2.K,
            self.cam2.D,
            self.cam1.xy,
            criteria=stereocalib_criteria,
            flags=flags,
        )
        self.ret = ret

    def _get_undistort_rectify_map(self):
        xy = self.cam1.xy
        self.R1, self.R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
            self.cam1.K, self.cam1.D, self.cam2.K, self.cam2.D, xy, self.R, self.t
        )
        self.undistort_rectify_map1 = cv2.initUndistortRectifyMap(
            self.cam1.K, self.cam1.D, self.R1, self.cam1.K, xy, cv2.CV_32FC1
        )
        self.undistort_rectify_map2 = cv2.initUndistortRectifyMap(
            self.cam2.K, self.cam2.D, self.R2, self.cam1.K, xy, cv2.CV_32FC1
        )

    def rectify(self, img1, img2):
        rectified1 = cv2.remap(
            self._get_img(img1),
            self.undistort_rectify_map1[0],
            self.undistort_rectify_map1[1],
            cv2.INTER_LANCZOS4,
        )
        rectified2 = cv2.remap(
            self._get_img(img2),
            self.undistort_rectify_map2[0],
            self.undistort_rectify_map2[1],
            cv2.INTER_LANCZOS4,
        )
        return [rectified1, rectified2]

    DUMP_ATTRS = ["R", "t"]

    def dump(self, path="", return_dict=False):

        dic = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in self.__dict__.items()
            if k in self.DUMP_ATTRS
        }
        dic["cam1"] = self.cam1.dump(return_dict=True)
        dic["cam2"] = self.cam2.dump(return_dict=True)
        if return_dict:
            return dic
        dic["_calibrating_version"] = __version__
        yamlstr = yaml.safe_dump(dic)
        if path:
            with open(path, "w") as f:
                f.write(yamlstr)
        return yamlstr

    def load(self, path_or_str_or_dict=None):
        from calibrating import Cam

        if path_or_str_or_dict is None:
            path_or_str_or_dict = self
            self = Stereo()
        if not isinstance(path_or_str_or_dict, (list, dict)):
            path_or_str = path_or_str_or_dict
            if "\n" in path_or_str:
                dic = yaml.safe_load(path_or_str)
            else:
                with open(path_or_str) as f:
                    dic = yaml.safe_load(f)
        else:
            dic = path_or_str_or_dict
        self.__dict__.update(
            {k: np.array(v) if k in self.DUMP_ATTRS else v for k, v in dic.items()}
        )
        self.cam1 = Cam.load(self.cam1)
        self.cam2 = Cam.load(self.cam2)
        self._get_undistort_rectify_map()
        return self

    @staticmethod
    def _get_img(path_or_np):
        if isinstance(path_or_np, str):
            return cv2.imread(path_or_np)[..., ::-1]
        return path_or_np

    vis = staticmethod(utils.vis_stereo)

    @classmethod
    def shows(cls, *l, **kv):
        vis = cls.vis(*l, **kv)

        idx = vis.shape[1] // 2
        viss = [vis[:, :idx], vis[:, idx:]]
        boxx.shows(viss)
        return viss

    def precision_analysis(self, depth_limits=(0.25, 5)):
        import matplotlib.pyplot as plt

        K = self.cam1.K
        baseline = self.baseline
        zmin, zmax = min(depth_limits), max(depth_limits)
        print(
            "-" * 15, "Stereo.precision_analysis", "-" * 15,
        )
        print("Baseline: %.2fcm" % (100 * baseline))
        print("cam1.xy:", self.cam1.xy)
        print("cam1.fovs:", utils._str_angle_dic(self.cam1.fovs))
        print("\n")
        dispmin = baseline * K[0, 0] / zmax
        dispmax = baseline * K[0, 0] / zmin
        disps = np.linspace(dispmin, dispmax)
        uncen_on_disp = -baseline * K[0, 0] / disps ** 2
        zs = baseline * K[0, 0] / disps
        disp_on_z = baseline * K[0, 0] / zs

        print("Depth - disparity 关系:")
        plt.plot(zs, disp_on_z)
        plt.grid()
        plt.show()
        print("Depth - 单位Disparity深度范围关系(深度不确定度):")
        plt.plot(zs, np.abs(uncen_on_disp))
        # plt.grid()
        # plt.show()
        # print("Depth - 单位像素在 xy 方向范围关系(xy不确定度):")
        # plt.plot(zs, zs / K[0, 0])
        plt.grid()
        plt.show()
        # TODO
        print("TODO: x,y 双目共同视野 - depth 关系:")
        print(
            "-" * 15, "End of Stereo.precision_analysis", "-" * 15,
        )

    MAX_DEPTH = 10

    def get_max_depth(self):
        return getattr(self, "max_depth", self.MAX_DEPTH)

    @property
    def baseline(self):
        return np.sum(self.t ** 2) ** 0.5

    def depth_to_disparity(self, depth):
        fx = self.cam1.K[0, 0]
        disparity = 1.0 * self.baseline * fx / depth
        return disparity

    def disparity_to_depth(self, disparity):
        fx = self.cam1.K[0, 0]
        depth = 1.0 * self.baseline * fx / disparity
        depth[depth > self.get_max_depth()] = 0
        return depth

    def unrectify_depth(self, depth):
        maps = cv2.initUndistortRectifyMap(
            self.cam1.K,
            None,
            np.linalg.inv(self.R1),
            self.cam1.K,
            self.cam1.xy,
            cv2.CV_32FC1,
        )

        ir = cv2.remap(depth, maps[0], maps[1], cv2.INTER_NEAREST)
        return ir

    def undistort_img(self, img1):
        return cv2.undistort(img1, self.cam1.K, self.cam1.D)

    def distort_depth(self, depth):
        """
        OOM warning and very slow
        """
        w, h = self.cam1.xy
        res = np.zeros((h, w), dtype=depth.dtype)

        ws = np.linspace(0, w, w, endpoint=False, dtype=np.int32)
        hs = np.linspace(0, h, h, endpoint=False, dtype=np.int32)
        u, v = np.meshgrid(ws, hs)
        u, v = u.reshape((-1, 1)), v.reshape((-1, 1))

        points = np.concatenate([u, v], axis=-1).astype(np.float32)
        depths = np.reshape(depth, (-1,))

        rtemp = ttemp = np.array([0, 0, 0], dtype=np.float32)

        undistort_points = cv2.undistortPoints(points, self.cam1.K, None)
        homogeneous_undistort_points = cv2.convertPointsToHomogeneous(undistort_points)

        imagePoints, _ = cv2.projectPoints(
            homogeneous_undistort_points,
            rtemp,
            ttemp,
            self.cam1.K,
            self.cam1.D,
            undistort_points,
        )
        imagePoints = np.reshape(imagePoints, (-1, 2)).astype(np.int32)
        points, index = np.unique(imagePoints, axis=0, return_index=True)
        res[points[:, 1], points[:, 0]] = depths[index]
        return res

    def set_stereo_matching(
        self, stereo_matching, max_depth=None, translation_rectify_img=False
    ):
        """
        Parameters
        ----------
        stereo_matching : MetaStereoMatching
            Subinstance of calibrating.MetaStereoMatching
        max_depth : float, optional
            The default is Stereo.MAX_DEPTH.
        translation_rectify_img : bool, the default is False.
            When self.get_depth(), translation rectify_img2 by self.min_disparity, 
            self.min_disparity is accroding to self.max_depth.
        """
        self.stereo_matching = stereo_matching
        self.translation_rectify_img = translation_rectify_img
        self.max_depth = max_depth or self.MAX_DEPTH
        self.min_disparity = int(self.cam1.K[0, 0] * self.baseline / self.max_depth)

    # TODO: return_unrectify_depth default False
    def get_depth(
        self, img1, img2, return_unrectify_depth=True, return_distort_depth=False
    ):
        """
        Return:
            rectify_img1
            rectify_depth
        The unit of depth is m
        """
        rectify_img1, rectify_img2 = self.rectify(img1, img2)
        if getattr(self, "translation_rectify_img"):
            rectify_img2[:, self.min_disparity :] = rectify_img2[
                :, : -self.min_disparity
            ]
            rectify_img2[:, : self.min_disparity] = 0
        # shows-(rectify_img1, rectify_img2)
        disparity = self.stereo_matching(rectify_img1, rectify_img2)
        if getattr(self, "translation_rectify_img"):
            disparity += self.min_disparity

        rectify_depth = self.disparity_to_depth(disparity)

        result = dict(
            rectify_img1=rectify_img1, rectify_depth=rectify_depth, disparity=disparity,
        )
        if return_distort_depth or return_unrectify_depth:
            unrectify_depth = self.unrectify_depth(rectify_depth)
            undistort_img1 = self.undistort_img(img1)
            result.update(
                unrectify_depth=unrectify_depth, undistort_img1=undistort_img1,
            )
        if return_distort_depth:
            result.update(
                distort_img1=img1, distort_depth=self.distort_depth(unrectify_depth),
            )
        return result


class MetaStereoMatching:
    def __init__(self, cfg=None):
        self.cfg = cfg

    def __call__(self, img1, img2):
        # input: RGB uint8 (h, w, 3)uint8
        raise NotImplementedError()
        # output: float disparity (h, w)float64, unit is m
        # return disparity


# A Example of StereoMatching class
class SemiGlobalBlockMatching(MetaStereoMatching):
    def __init__(self, cfg):
        default_cfg = {
            "max_size": 1000,
            # SGBM_args
            "P1": 4590,
            "P2": 18360,
            "blockSize": 5,
            "disp12MaxDiff": 1,
            "minDisparity": 2,
            "mode": 2,
            "numDisparities": 80,
            "preFilterCap": 63,
            "speckleRange": 2,
            "speckleWindowSize": 0,
            "uniquenessRatio": 15,
        }
        default_cfg.update(cfg)
        self.cfg = default_cfg
        self.max_size = self.cfg.pop("max_size")
        self.stereo_sgbm = cv2.StereoSGBM_create(**self.cfg)

    def __call__(self, img1, img2):
        resize_ratio = min(self.max_size / max(img1.shape[:2]), 1)
        simg1, simg2 = boxx.resize(img1, resize_ratio), boxx.resize(img2, resize_ratio)
        sdisparity = (self.stereo_sgbm.compute(simg1, simg2).astype(np.float32)).clip(
            0
        ) / 16.0
        disparity = boxx.resize(sdisparity, 1 / resize_ratio) / resize_ratio
        return disparity


if __name__ == "__main__":
    from boxx import *
    from calibrating import Cam

    cam1, cam2, camd = Cam.get_test_cams()

    stereo = Stereo(cam1, cam2)

    yaml_path = "/tmp/stereo.yaml"
    stereo.dump(yaml_path)
    stereo = Stereo.load(yaml_path)

    cam1.vis_stereo(cam2, stereo)

    stereo_matching = SemiGlobalBlockMatching({})
    stereo.set_stereo_matching(
        stereo_matching, max_depth=3.5, translation_rectify_img=True
    )

    key = list(cam1)[0]
    img1 = boxx.imread(cam1[key]["path"])
    img2 = boxx.imread(cam2[key]["path"])

    re = stereo.get_depth(img1, img2)
    boxx.tree(re)
    utils.vis_align(re["undistort_img1"], re["unrectify_depth"])
