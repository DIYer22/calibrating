#!/usr/bin/env python3
import cv2
import yaml
import boxx
import numpy as np


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

    def __init__(self, cam1=None, cam2=None, force_same_intrinsic=True):
        if cam1 is None:
            return
        self.cam1 = cam1
        self.cam2 = cam2
        self.force_same_intrinsic = force_same_intrinsic
        self.get_R_t_by_stereo_calibrate()
        self._get_undistort_rectify_map()

    def get_R_t_by_stereo_calibrate(self):
        keys = self.cam1.valid_keys_intersection(self.cam2)
        image_pointsl = [self.cam1[key]["image_points"] for key in keys]
        image_pointsr = [self.cam2[key]["image_points"] for key in keys]
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
            image_pointsl,
            image_pointsr,
            self.cam1.K,
            self.cam1.D,
            self.cam2.K,
            self.cam2.K,
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
            self.cam2.K, self.cam2.D, self.R2, self.cam2.K, xy, cv2.CV_32FC1
        )

    def rectify(self, img1, img2):
        ir1 = cv2.remap(
            self._get_img(img1),
            self.undistort_rectify_map1[0],
            self.undistort_rectify_map1[1],
            cv2.INTER_LANCZOS4,
        )
        ir2 = cv2.remap(
            self._get_img(img2),
            self.undistort_rectify_map2[0],
            self.undistort_rectify_map2[1],
            cv2.INTER_LANCZOS4,
        )
        return [ir1, ir2]

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

    @staticmethod
    def vis(img1, img2, n_line=21, thickness=0.03):
        """
        Draw lines on stereo image pairs, two cases of horizontal rectify and vertical rectify.
        Input: image numpy pairs.
        """
        colors = [
            [255, 0, 0],
            [0, 255, 255],
            [0, 255, 0],
            [255, 0, 255],
            [0, 0, 255],
            [255, 255, 0],
        ]
        vis = np.concatenate([img1, img2], 1)
        img_size = img1.shape[0]
        _thickness = max(1, int(round(thickness * img_size / (n_line + 1))))
        gap = img_size / (n_line + 1)
        for i in range(n_line):
            b = int((i + 1) * gap - _thickness / 2)
            vis[b : b + _thickness] = colors[i % len(colors)]
        return vis

    @classmethod
    def shows(cls, *l, **kv):
        vis = cls.vis(*l, **kv)

        idx = vis.shape[1] // 2
        viss = [vis[:, :idx], vis[:, idx:]]
        boxx.shows(viss)
        return viss

    # TODO
    def depth_to_disparity(self, depth):
        """
        Used for genrate GT of disparity
        """
        raise NotImplementedError()

    def disparity_to_depth(self, disparity):
        raise NotImplementedError()

    def unrectify_depth(self, depth):
        raise NotImplementedError()

    def undistort_img(self, img1):
        return cv2.undistort(img1, self.cam1.K, self.cam1.D)

    def distort_depth(self, depth):
        raise NotImplementedError()

    def set_stereo_matching(self, stereo_matching):
        self.stereo_matching = stereo_matching

    def get_depth(self, img1, img2):
        disparity = self.stereo_matching(img1, img2)
        rectify_depth = self.disparity_to_depth(disparity)
        unrectify_depth = self.unrectify_depth(rectify_depth)
        undistort_img1 = self.undistort_img(img1)
        return dict(
            disparity=disparity,
            rectify_depth=rectify_depth,
            unrectify_depth=unrectify_depth,
            undistort_img1=undistort_img1,
        )


class MetaStereoMatching:
    def __init__(self, cfg=None):
        self.cfg = cfg

    def __call__(self, img1, img2):
        # input: RGB uint8 (h, w, 3)uint8
        # output: float disparity (h, w)float64
        return img1.mean(-1)


if __name__ == "__main__":
    from calibrating import *
    from calibrating import Cam

    cam1, cam2, camd = Cam.get_test_cams()

    stereo = Stereo(cam1, cam2)

    yaml_path = "/tmp/stereo.yaml"
    stereo.dump(yaml_path)
    stereo = Stereo.load(yaml_path)

    cam1.vis_stereo(cam2, stereo)

    stereo_matching = MetaStereoMatching({})
    stereo.set_stereo_matching(stereo_matching)

    key = list(cam1)[0]
    img1 = boxx.imread(cam1[key]["path"])
    img2 = boxx.imread(cam2[key]["path"])

    re = stereo.get_depth(img1, img2)
    boxx.tree(re)
    boxx.shows(re)
