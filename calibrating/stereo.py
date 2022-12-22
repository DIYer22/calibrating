#!/usr/bin/env python3
import cv2
import yaml
import boxx
import numpy as np
from functools import wraps

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

    def __init__(
        self,
        cam1=None,
        cam2=None,
        force_same_intrinsic=False,
        xy_target=None,
        K_target=1,
    ):
        self.xy_target = xy_target
        self.K_target = K_target
        if cam1 is None:
            return
        self.cam1 = cam1
        self.cam2 = cam2
        self.force_same_intrinsic = force_same_intrinsic
        # TODO better cx cy
        self.get_R_t_by_stereo_calibrate()
        self._get_undistort_rectify_map()

    def get_conjoint_points(self):
        image_points = list(self.cam1.values())[0]["image_points"]
        # "image_points" is checkboard type: shape(n, 2)
        if isinstance(image_points, np.ndarray):
            keys = self.cam1.valid_keys_intersection(self.cam2)
            conjoint_image_points1 = [self.cam1[key]["image_points"] for key in keys]
            conjoint_image_points2 = [self.cam2[key]["image_points"] for key in keys]
            conjoint_object_points = [self.cam1[key]["object_points"] for key in keys]

        # "image_points" is marker type: {id: shape(n, 2)}
        elif isinstance(image_points, dict):
            conjoint_image_points1, conjoint_image_points2, conjoint_object_points = (
                [],
                [],
                [],
            )
            for key in self.cam1:
                a, b, c = [], [], []
                for id in self.cam1[key].get("image_points", {}):
                    if key in self.cam2 and id in self.cam2[key].get(
                        "image_points", {}
                    ):
                        a.append(self.cam1[key]["image_points"][id])
                        b.append(self.cam2[key]["image_points"][id])
                        c.append(self.cam1[key]["object_points"][id])
                if a:
                    conjoint_image_points1.append(np.concatenate(a, 0))
                    conjoint_image_points2.append(np.concatenate(b, 0))
                    conjoint_object_points.append(np.concatenate(c, 0))
        return (
            conjoint_image_points1,
            conjoint_image_points2,
            conjoint_object_points,
        )

    def get_R_t_by_stereo_calibrate(self):

        (
            conjoint_image_points1,
            conjoint_image_points2,
            conjoint_object_points,
        ) = self.get_conjoint_points()

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
        retval, _, _, _, _, self.R, self.t, E, F = cv2.stereoCalibrate(
            conjoint_object_points,
            conjoint_image_points1,
            conjoint_image_points2,
            self.cam1.K,
            self.cam1.D,
            self.cam2.K,
            self.cam2.D,
            self.cam1.xy,
            flags=flags,
            criteria=stereocalib_criteria,
        )
        self.retval = retval

    def _get_undistort_rectify_map(self):

        if self.xy_target is None:
            self.xy_target = self.cam1.xy
        if isinstance(self.xy_target, (int, float)):
            self.xy_target = [int(round(i * self.xy_target)) for i in self.cam1.xy]
        xy = tuple(self.xy_target)
        if isinstance(self.K_target, (int, float)):
            K_target = self.K_target
            self.K_target = self.cam1.K.copy()
            self.K_target[:2, :2] *= K_target
            self.K_target[:2, 2] += (np.array(xy) - self.cam1.xy) / 2

        self.stereo_recitfy()

        self.undistort_rectify_map1 = cv2.initUndistortRectifyMap(
            self.cam1.K, self.cam1.D, self.R1, self.K_target, xy, cv2.CV_32FC1
        )

        self.undistort_rectify_map2 = cv2.initUndistortRectifyMap(
            self.cam2.K, self.cam2.D, self.R2, self.K_target, xy, cv2.CV_32FC1
        )
        assert (utils.R_t_to_T(self.R2.T) @ utils.R_t_to_T(self.R, self.t))[
            0, 3
        ] < 0, "The left and right cameras may be wrong. Please try to switch the camera order of Stereo(cam1, cam2)"

    def stereo_recitfy_by_cv2(self):
        """When the right eye camera is rotated 180° around the z axis, a BUG will appear"""
        self.R1, self.R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
            self.cam1.K,
            self.cam1.D,
            self.cam2.K,
            self.cam2.D,
            tuple(self.xy_target),
            self.R,
            self.t,
        )

    def stereo_recitfy(self):
        axes_z = np.array([0, 0, 1.0])
        axes_nx = np.array([-1.0, 0, 0])
        t = self.t.squeeze()
        plane_v = t
        z_on_plane2 = utils.project_vec_on_plane(axes_z, plane_v)
        z_on_plane1 = utils.project_vec_on_plane(self.R @ axes_z, plane_v)
        z_on_plane = z_on_plane2 / np.linalg.norm(
            z_on_plane2
        ) + z_on_plane1 / np.linalg.norm(z_on_plane1)

        R_align_x = utils.rotate_shortest_of_two_vecs(axes_nx, t)
        R_align_z = utils.rotate_shortest_of_two_vecs(R_align_x @ axes_z, z_on_plane)

        self.R2 = (R_align_z @ R_align_x).T
        self.R1 = self.R2 @ self.R[:3, :3]

    def rectify(self, img1, img2):
        rectify_img1 = cv2.remap(
            self._get_img(img1),
            self.undistort_rectify_map1[0],
            self.undistort_rectify_map1[1],
            cv2.INTER_LANCZOS4,
        )
        rectify_img2 = cv2.remap(
            self._get_img(img2),
            self.undistort_rectify_map2[0],
            self.undistort_rectify_map2[1],
            cv2.INTER_LANCZOS4,
        )

        if getattr(self, "translation_rectify_img", None):
            if self.min_disparity > 0:
                rectify_img2[:, self.min_disparity :] = rectify_img2[
                    :, : -self.min_disparity
                ]
                rectify_img2[:, : self.min_disparity] = 0
            elif self.min_disparity < 0:
                rectify_img2[:, : self.min_disparity] = rectify_img2[
                    :, -self.min_disparity :
                ]
                rectify_img2[:, self.min_disparity :] = 0

        return [rectify_img1, rectify_img2]

    DUMP_ATTRS = ["R", "t", "retval"]

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
            dic = dict(path_or_str_or_dict)

        if hasattr(self, "cam1"):
            self.cam1.load(dic.pop("cam1"))
            self.cam2.load(dic.pop("cam2"))
        else:
            self.cam1 = Cam.load(dic.pop("cam1"))
            self.cam2 = Cam.load(dic.pop("cam2"))
        self.__dict__.update(
            {k: np.array(v) if k in self.DUMP_ATTRS else v for k, v in dic.items()}
        )
        self._get_undistort_rectify_map()
        return self

    def copy(self):
        new = type(self)()
        new.load(self.dump())
        return new

    @staticmethod
    def _get_img(path_or_np):
        if isinstance(path_or_np, str):
            return cv2.imread(path_or_np)[..., ::-1]
        return path_or_np

    @wraps(utils.vis_stereo)
    def vis(self, *args, **argkws):
        if not isinstance(self, Stereo):
            args = (self,) + args
        if not args:
            key = list(self.cam1)[0]
            img1 = boxx.imread(self.cam1[key]["path"])
            img2 = boxx.imread(self.cam2[key]["path"])
            args = self.rectify(img1, img2)
        return utils.vis_stereo(*args, **argkws)

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
        print("-" * 15, "Stereo.precision_analysis", "-" * 15)
        print(self)
        print("\n")
        dispmin = baseline * K[0, 0] / zmax
        dispmax = baseline * K[0, 0] / zmin
        disps = np.linspace(dispmin, dispmax)
        uncen_on_disp = -baseline * K[0, 0] / disps**2
        zs = baseline * K[0, 0] / disps
        disp_on_z = baseline * K[0, 0] / zs

        print("Depth(m) - disparity(pixel):")
        plt.plot(zs, disp_on_z)
        plt.grid()
        plt.show()
        print("\nTheoretical zmap precision:")
        print(
            "\tDepth - The depth range represented by a pixel disparity(depth uncertainty)"
        )
        plt.plot(zs, np.abs(uncen_on_disp))
        # plt.grid()
        # plt.show()
        # print("Depth - 单位像素在 xy 方向范围关系(xy不确定度):")
        # plt.plot(zs, zs / K[0, 0])
        plt.grid()
        plt.show()
        # TODO: x,y 双目共同视野 - depth 关系
        print("-" * 15, "End of Stereo.precision_analysis", "-" * 15)

    def __str__(self):
        r = utils.T_to_r(self.R).squeeze()
        du = np.linalg.norm(r) * 180 / np.pi
        try:
            strr = "Stereo(cam1='%s', cam2='%s'):\n" % (self.cam1.name, self.cam2.name)
            strr += "\t xy: %s\n" % str(list(self.cam1.xy))[1:-1]
            strr += "\t baseline: %.2fcm\n" % (100 * self.baseline)
            strr += "\t t(cm): [%s]\n" % (
                " ".join([str(i) for i in (self.t.squeeze() * 100).round(2)])
            )
            strr += "\t r(rodrigues): [%s] %.2f°\n" % (
                " ".join([str(i) for i in r.round(3)]),
                du,
            )
            strr += "\t cam1.fovs: %s\n" % str(utils._str_angle_dic(self.cam1.fovs))
            if hasattr(self, "retval"):
                strr += "\t retval: %s\n" % self.retval
        except Exception as e:
            strr += f"\t Exception({e}) in Stereo.__str__()"
        return strr

    __repr__ = __str__

    MAX_DEPTH = 10

    def get_max_depth(self):
        return getattr(self, "max_depth", self.MAX_DEPTH)

    @property
    def xy(self):
        return self.cam1.xy

    @property
    def K(self):
        return self.cam1.K

    @property
    def D(self):
        return np.zeros((1, 5))

    @property
    def baseline(self):
        return np.sum(self.t**2) ** 0.5

    def depth_to_disparity(self, depth):
        fx = self.K_target[0, 0]
        disparity = 1.0 * self.baseline * fx / depth
        return disparity

    def disparity_to_depth(self, disparity):
        fx = self.K_target[0, 0]
        depth = 1.0 * self.baseline * fx / disparity
        depth[depth > self.get_max_depth()] = 0
        return depth

    def unrectify_depth(self, depth):
        maps = getattr(self, "_unrectify_depth_maps", None)
        re = utils.rotate_depth_by_remap(
            self.K_target,
            self.R1.T,
            depth,
            maps=maps,
            xy_target=self.cam1.xy,
            K_target=self.cam1.K,
        )
        if maps is None:
            self._unrectify_depth_maps = re["maps"]
        rotated_depth = re["depth"]
        return rotated_depth

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
        self, stereo_matching, max_depth=None, translation_rectify_img=None
    ):
        """
        Parameters
        ----------
        stereo_matching : MetaStereoMatching
            Subinstance of calibrating.MetaStereoMatching
        max_depth : float, optional
            The default is Stereo.MAX_DEPTH.
        translation_rectify_img : bool, the default is None.
            When self.get_depth(), translation rectify_img2 by self.min_disparity,
            self.min_disparity is accroding to self.max_depth.
            default is accroding to bool(max_depth)
        """
        self.stereo_matching = stereo_matching
        self.translation_rectify_img = (
            bool(max_depth)
            if translation_rectify_img is None
            else translation_rectify_img
        )
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

        assert hasattr(
            self, "stereo_matching"
        ), "Please stereo.set_stereo_matching(stereo_matching)"
        rectify_img1, rectify_img2 = self.rectify(img1, img2)
        disparity = self.stereo_matching(rectify_img1, rectify_img2)
        if getattr(self, "translation_rectify_img"):
            disparity += self.min_disparity

        rectify_depth = self.disparity_to_depth(disparity)

        result = dict(
            rectify_img1=rectify_img1,
            rectify_depth=rectify_depth,
            disparity=disparity,
            rectify_img2=rectify_img2,
        )
        if return_distort_depth or return_unrectify_depth:
            unrectify_depth = self.unrectify_depth(rectify_depth)
            undistort_img1 = self.undistort_img(img1)
            result.update(
                unrectify_depth=unrectify_depth,
                undistort_img1=undistort_img1,
            )
        if return_distort_depth:
            result.update(
                distort_img1=img1,
                distort_depth=self.distort_depth(unrectify_depth),
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
    def __init__(self, cfg=None):
        if cfg is None:
            cfg = {}
        self.cfg = cfg
        self.max_size = self.cfg.get("max_size", 1000)

        # StereoSGBM_create from https://gist.github.com/andijakl/ffe6e5e16742455291ef2a4edbe63cb7
        block_size = 11
        min_disp = 2
        max_disp = 220
        # Maximum disparity minus minimum disparity. The value is always greater than zero.
        # In the current implementation, this parameter must be divisible by 16.
        num_disp = max_disp - min_disp
        # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
        # Normally, a value within the 5-15 range is good enough
        uniquenessRatio = 5
        # Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
        # Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
        speckleWindowSize = 200
        # Maximum disparity variation within each connected component.
        # If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
        # Normally, 1 or 2 is good enough.
        speckleRange = 2
        disp12MaxDiff = 0

        self.stereo_sgbm = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            uniquenessRatio=uniquenessRatio,
            speckleWindowSize=speckleWindowSize,
            speckleRange=speckleRange,
            disp12MaxDiff=disp12MaxDiff,
            P1=8 * 1 * block_size * block_size,
            P2=32 * 1 * block_size * block_size,
        )

    def __call__(self, img1, img2):
        resize_ratio = min(self.max_size / max(img1.shape[:2]), 1)
        simg1, simg2 = boxx.resize(img1, resize_ratio), boxx.resize(img2, resize_ratio)
        sdisparity = (self.stereo_sgbm.compute(simg1, simg2).astype(np.float32)).clip(0)
        sdisparity[sdisparity < self.stereo_sgbm.getMinDisparity() * 16] = 0
        disparity = (
            boxx.resize(sdisparity / 16.0, img1.shape[:2])
            * img1.shape[1]
            / simg1.shape[1]
        )
        return disparity


class MatchingByBoard(MetaStereoMatching):
    """
    Stereo mathch disp by board's image_points
    return a dense disp on calibration board
    """

    def __init__(self, board, dense_predict=True):
        self.board = board
        self.dense_predict = dense_predict

    def __call__(self, img1, img2):
        self.board
        d1 = dict(img=img1)
        self.board.find_image_points(d1)
        image_points1 = d1["image_points"]
        d2 = dict(img=img2)
        self.board.find_image_points(d2)
        image_points2 = d2["image_points"]

        if isinstance(image_points1, dict):
            image_points1 = []
            image_points2 = []
            for key in sorted(set(d1["image_points"]).intersection(d2["image_points"])):
                image_points1.append(d1["image_points"][key])
                image_points2.append(d2["image_points"][key])
            image_points1 = np.concatenate(image_points1, 0)
            image_points2 = np.concatenate(image_points2, 0)

        rectify_std = np.std((image_points2 - image_points1)[:, 1])
        point_disps = (image_points1 - image_points2)[:, 0]
        xyds = np.append(image_points1, point_disps[:, None], axis=-1)
        sparse_disp = utils.uvzs_to_arr2d(xyds, img1.shape[:2])
        if self.dense_predict:
            dense_disp = utils.interpolate_sparse2d(sparse_disp, "convex_hull")
            return dense_disp
        else:
            return sparse_disp


if __name__ == "__main__":
    from boxx import *
    from calibrating import Cam

    cam1, cam2, camd = Cam.get_test_cams()

    stereo = Stereo(cam1, cam2)
    print(stereo)

    yaml_path = "/tmp/stereo.yaml"
    stereo.dump(yaml_path)
    stereo = Stereo.load(yaml_path)

    cam1.vis_stereo(cam2, stereo)

    stereo_matching = SemiGlobalBlockMatching({})
    stereo.set_stereo_matching(stereo_matching, max_depth=3.5)

    key = list(cam1)[0]
    img1 = boxx.imread(cam1[key]["path"])
    img2 = boxx.imread(cam2[key]["path"])

    re = stereo.get_depth(img1, img2)
    boxx.tree(re)
    utils.vis_align(re["undistort_img1"], re["unrectify_depth"])
    stereo.precision_analysis()
