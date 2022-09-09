#!/usr/bin/env python3

import os
import cv2
import boxx
import numpy as np
from glob import glob

from boxx import imread, shows, show, showb


def R_t_to_T(R, t):
    T = np.zeros((4, 4))
    T[:3, :3] = R
    T[:3, -1] = t.squeeze()
    T[3, 3] = 1
    return T


def r_t_to_T(r, t):
    assert r.size == 3
    return R_t_to_T(cv2.Rodrigues(r.squeeze())[0], t.squeeze())


def T_to_r(T):
    rvec = cv2.Rodrigues(T[:3, :3])[0]
    return rvec


def T_to_r_t(T):
    rvec = T_to_r(T)
    tvec = T[:3, 3:]
    if not tvec.size:
        tvec = np.zeros((3, 1))
    return rvec, tvec


def R_to_deg(r_R_T):
    if r_R_T.size == 3:
        r = r_R_T
    elif r_R_T.size >= 9:
        r = T_to_r(r_R_T).squeeze()
    deg = np.linalg.norm(r) * 180 / np.pi
    return deg


def T_to_deg_distance(T, compare_T=None):
    """
    Translate the T matrix into items that humans can intuitively understand, 
    such as degrees of rotation and displacement distances
    Set compare_T to compare the difference between two Ts

    Returns
    -------
    └── /: dict  4
        ├── deg: 45.2°
        ├── distance: 0.2 m
        ├── r: (3, 1)float64
        └── t: (3, 1)float64
    """
    if compare_T is not None:
        T = np.linalg.inv(compare_T) @ T
    r, t = T_to_r_t(T)
    deg = R_to_deg(r)
    distance = np.linalg.norm(t)
    return dict(deg=deg, distance=distance, r=r, t=t)


def convert_points_for_cv2(dic_or_np):
    point = dic_or_np
    if isinstance(point, dict):
        point = np.concatenate([point[id] for id in sorted(point)], axis=0)
    return point


def apply_T_to_point_cloud(T, point_cloud):
    new_point_cloud = np.ones((len(point_cloud), 4))
    new_point_cloud[:, :3] = point_cloud
    return (T @ new_point_cloud.T).T[:, :3]


def rotate_depth_by_point_cloud(K, R, depth, interpolation_rate=1):
    pc = depth_to_point_cloud(depth, K, interpolation_rate=interpolation_rate)
    T = np.eye(4)
    T[:3, :3] = R
    new_pc = apply_T_to_point_cloud(T, pc)
    new_depth = point_cloud_to_depth(new_pc, K, depth.shape[::-1])
    return dict(depth=new_depth, R=R)


def rotate_depth_by_remap(K, R, depth, maps=None):
    y, x = depth.shape
    if maps is None:
        maps = cv2.initUndistortRectifyMap(K, None, R, K, (x, y), cv2.CV_32FC1,)
    ys, xs = np.mgrid[:y, :x]
    ys, xs = ys.flatten(), xs.flatten()
    points = np.array([xs, ys, np.ones_like(xs)]) * depth.flatten()[None]
    # points.shape is (3, n)
    new_points = (R @ np.linalg.inv(K) @ points).T
    new_depth_on_old_xy = new_points[:, 2].reshape(y, x)

    new_depth = cv2.remap(new_depth_on_old_xy, maps[0], maps[1], cv2.INTER_NEAREST)
    return dict(depth=new_depth, R=R, maps=maps)


def depth_to_point_cloud(depth, K, interpolation_rate=1):
    # depth to point cloud
    # K@Ps/z=xy
    assert depth.ndim == 2
    y, x = depth.shape
    if depth.dtype == np.uint16:
        depth = np.float32(depth / 1000.0)
    if interpolation_rate == 1:
        ys, xs = np.mgrid[
            :y, :x,
        ]

        points = (np.array([xs, ys, np.ones_like(xs)]) * depth[None])[:, depth != 0].T
    else:
        y_, x_ = int(round(y * interpolation_rate)), int(round(x * interpolation_rate))
        depth_ = cv2.resize(depth, (x_, y_), interpolation=cv2.INTER_NEAREST)
        ys, xs = np.mgrid[
            :y_, :x_,
        ]
        points = (np.array([xs, ys, np.ones_like(xs)]) * depth_[None])[:, depth_ != 0].T
        points[:, 0] /= interpolation_rate
        points[:, 1] /= interpolation_rate

    point_cloud = (np.linalg.inv(K) @ points.T).T
    return point_cloud


def point_cloud_to_depth(points, K, xy):
    xyzs = (K @ points.T).T
    xyzs[:, :2] /= xyzs[:, 2:]
    sorted_idx = np.argsort(-xyzs[:, 2])
    xyzs = xyzs[sorted_idx]
    return uvzs_to_arr2d(xyzs, xy[::-1])


def uvzs_to_arr2d(uvzs, hw=None, bg_value=0, arr2d=None):
    if arr2d is None:
        if hw is None:
            hw = np.int32(uvzs.max(0)[:2].round()) + 1
        arr2d = np.ones(hw, uvzs.dtype) * bg_value
    else:
        hw = arr2d.shape[:2]

    xs, ys = np.int32(uvzs[:, :2].round()).T
    mask = (xs >= 0) & (xs < hw[1]) & (ys >= 0) & (ys < hw[0])
    arr2d[ys[mask], xs[mask]] = uvzs[:, 2][mask]
    return arr2d


# TODO rm
def xyzs_to_arr2d(*args, **argkws):
    print("\n\nWarning: xyzs_to_arr2d have rename to uvzs_to_arr2d")
    return uvzs_to_arr2d(*args, **argkws)


def arr2d_to_uvzs(arr2d, mask=None):
    y, x = arr2d.shape
    ys, xs = np.mgrid[:y, :x]
    if mask is None:
        uvzs = np.array([xs, ys, arr2d]).T.reshape(-1, 3)
    else:
        if mask.dtype != np.bool8:
            mask = np.bool8(mask)
        uvzs = np.array([xs[mask], ys[mask], arr2d[mask]]).T
    return uvzs


def interpolate_sparse2d(sparse2d, constrained_type=None, inter_type="lstsq"):
    """
    if constrained_type == True, then only interpolate convexHull area
    """
    mask_to_uvzs = (sparse2d != 0) & np.isfinite(sparse2d)
    uvzs = arr2d_to_uvzs(sparse2d, mask_to_uvzs)
    input_mask = np.zeros_like(sparse2d, np.uint8)
    if constrained_type is not None and constrained_type:
        convex_hull = cv2.convexHull(np.int32(uvzs[:, :2].round()))
        cv2.drawContours(input_mask, [convex_hull], -1, 1, -1)
        input_uvs = arr2d_to_uvzs(input_mask, input_mask)
    else:
        input_uvs = arr2d_to_uvzs(input_mask)

    # TODO replaced by Solving equations of the z = ay+bx+c
    # fit uvzs linearly
    if inter_type == "rbf":
        import scipy.interpolate

        spline = scipy.interpolate.Rbf(
            uvzs[:, 0],
            uvzs[:, 1],
            uvzs[:, 2],
            function="thin_plate",
            smooth=0.5,
            episilon=5,
        )
        output = spline(input_uvs[:, 0], input_uvs[:, 1],)
        output_uvzs = np.append(input_uvs, output[:, None], axis=-1)
    elif inter_type == "lstsq":
        A = uvzs.copy()
        A[:, 2] = 1
        lstsq_re = np.linalg.lstsq(A, uvzs[:, 2], rcond=None)
        abc = lstsq_re[0]
        output_uvzs = np.float32(input_uvs)
        output_uvzs[:, 2] = 1
        output_uvzs[:, 2] = output_uvzs @ abc

    dense = uvzs_to_arr2d(output_uvzs, sparse2d.shape, arr2d=sparse2d.copy())
    return dense


def mean_Ts(Ts):
    Ts = np.array(Ts)
    Rs = Ts[:, :3, :3]
    R0 = Rs[0]
    for idx, R in enumerate(Rs):
        if not idx:
            R_acc = R0.T @ R
        else:
            R_acc = (R0.T @ R) @ R_acc
    r_acc = T_to_r(R_acc)
    r_mean = r_acc / len(Ts)
    R_mean = R0 @ cv2.Rodrigues(r_mean)[0]

    T = R_t_to_T(R_mean, np.mean(Ts, 0)[:3, 3])
    return T


def intrinsic_format_conversion(K_or_dic):
    if isinstance(K_or_dic, dict):
        dic = K_or_dic
        return np.float64(
            [[dic["fx"], 0, dic["cx"]], [0, dic["fy"], dic["cy"]], [0, 0, 1]]
        )
    else:
        dic, K = {}, K_or_dic.tolist()
        dic["fx"] = K[0][0]
        dic["cx"] = K[0][2]
        dic["fy"] = K[1][1]
        dic["cy"] = K[1][2]
        return dic


def _str_angle_dic(angle_dic):
    return ", ".join(["%s=%s°" % (k, round(v, 2)) for k, v in angle_dic.items()])


def _to_3x_uint8(arr):
    if arr.dtype != np.uint8:
        arr = boxx.uint8(boxx.norma(arr))
        arr = np.uint8(cv2.applyColorMap(arr, cv2.COLORMAP_JET) * 0.75)
    elif arr.ndim == 2:
        arr = np.transpose([arr] * 3, (1, 2, 0))
    return arr


def vis_depth(depth, slicen=0, fix_range=None, colormap=None):
    if depth.ndim == 3 and depth.shape[-1] in [3, 4]:
        return depth
    if depth.dtype == np.uint16:
        depth = depth / 1000.0
    raw_depth = depth
    if fix_range:
        if isinstance(fix_range, (float, int)):
            fix_range = (0, fix_range)
        depth = depth.clip(*fix_range)
        normaed = (depth - fix_range[0]) / (fix_range[1] - fix_range[0])
    else:
        normaed = boxx.norma(depth)
    if slicen:
        normaed = (normaed * slicen) % 1
        if colormap is None:
            colormap = cv2.COLORMAP_HSV
    depth_uint8 = np.uint8(normaed * 255.9)
    vis = cv2.applyColorMap(depth_uint8, colormap or cv2.COLORMAP_JET)[..., ::-1]
    vis[raw_depth == 0] = 0
    return vis


def vis_depth_l1(re, gt=0, max_l1=0.05, overexposed=True):
    """
    Pretty vis of depth l1, which will ignore missing depth.
    Could distinguish missing depth(black) and l1==0(grey)
    For l1>0(far) red, l1<0(near) green, color of overexposed area will turns white
    
    Parameters
    ----------
    re : depth
        unit is m
    gt : depth or float, optional
        The default is 0.
    max_l1 : float, optional
        max l1 to vis. The default is 0.05.
    """
    GREY_CLIP = 0.1  # how_grey_to_distinguish missing depth and l1==0
    if isinstance(gt, (int, float, np.number)):
        gt = np.ones_like(re) * gt

    mask_valid = np.bool8(re) & np.bool8(gt)
    l1 = (re - gt) * mask_valid
    l1_gt_0 = l1 > 0
    l1_lt_0 = l1 < 0
    # l1>0(far) red, l1<0(near) green
    l1_vis_ = np.array([l1 * (l1_gt_0), -l1 * (l1_lt_0), l1 * 0])
    l1_vis_norma = l1_vis_.clip(0, max_l1) / max_l1
    l1_vis_with_grey = (l1_vis_norma * (1 - GREY_CLIP) + GREY_CLIP) * mask_valid
    l1_vis = np.uint8(l1_vis_with_grey * 255).transpose(1, 2, 0)
    if overexposed:
        overexposed_mask = np.abs(l1) > max_l1
        l1_vis[overexposed_mask & (l1_gt_0), 1:] = [[255, 0]]
        l1_vis[overexposed_mask & (l1_lt_0), ::2] = 230
    return l1_vis


def vis_point_uvs(
    uvs,
    img_or_shape=None,
    size=None,
    color=None,
    contour=False,
    convex_hull=False,
    full_connect=False,
):
    if uvs.ndim == 3 and uvs.shape[1] == 1:
        uvs = uvs[:, 0]
    uvs = np.int32(uvs.round())
    if isinstance(img_or_shape, np.ndarray) and img_or_shape.ndim >= 2:
        vis = img_or_shape.copy()
    else:
        if img_or_shape is None:
            img_or_shape = [int(uvs[:, :2].max() * 1.1)] * 2
        vis = np.ones(img_or_shape, np.uint8) * 128
    if vis.ndim == 2:
        vis = np.concatenate([vis[:, :, None]] * 3, -1)
    if size is None:
        size = 1.0
    if size < 5 and isinstance(size, float):
        size = np.sum(vis.shape[:2]) * 0.001 * size
        size = max(1, size)
    size = int(round(size))
    _color = (255, 0, 0) if color is None else color
    _draw_contour = False
    if convex_hull:
        contour_points = cv2.convexHull(uvs)
        _draw_contour = True
    if contour:
        contour_points = uvs
        _draw_contour = True
    if _draw_contour:
        cv2.drawContours(
            vis, [np.int32(contour_points.round())], 0, _color, size * 2 // 3
        )
    if full_connect:
        for i in range(len(uvs) - 1):
            for j in range(i, len(uvs)):
                cv2.line(vis, uvs[i], uvs[j], _color, size * 2 // 3)
    for idx, uv in enumerate(uvs):
        if color is None:
            vis = cv2.circle(vis, tuple(uv[:2]), size * 2, (255, 255, 255), -1)
        vis = cv2.circle(vis, tuple(uv[:2]), size, _color, -1,)
    return vis


def _get_vis_background_of_cam(cam):
    # vis = np.ones(list(cam.xy[::-1]) + [3], np.uint8) * 128
    ys, xs = np.mgrid[: cam.xy[1], : cam.xy[0]]
    bg = (
        ((ys - cam.xy[1] / 2) ** 2 + (xs - cam.xy[0] / 2) ** 2 + 1e-6) ** 0.5
        / (np.mean(cam.xy) / 2 / 5)
        % 1
    )
    bg = cv2.cvtColor(boxx.uint8(bg) // 4 + 63, cv2.COLOR_GRAY2RGB)
    return bg


def vis_T(T, cam=None, img=None, length=0.1):
    if cam is None:
        from calibrating import Cam

        cam = Cam.get_example_720p()
    if img is None:
        vis = _get_vis_background_of_cam(cam)
    else:
        vis = img.copy()
    rvec, tvec = T_to_r_t(T)
    cv2.drawFrameAxes(vis, cam.K, cam.D, rvec, tvec, length)
    return vis


def vis_stereo(img1, img2, n_line=21, thickness=0.03):
    """
    Draw lines on stereo image pairs, two cases of horizontal rectify and vertical rectify.
    Input: image numpy pairs.
    """
    img1 = _to_3x_uint8(img1)
    img2 = _to_3x_uint8(img2)
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


def vis_align(img1, img2, n_line=21, shows=True):
    img1 = _to_3x_uint8(img1)
    img2 = _to_3x_uint8(img2)
    y, x = img1.shape[:2]
    visv = vis_stereo(img1, img2, n_line=n_line)
    vis = np.concatenate((visv, vis_stereo(img2, img1, n_line=n_line),), 0)
    vis = np.rot90(
        vis_stereo(
            np.rot90(vis[:y]), np.rot90(vis[y:]), n_line=int(n_line * x * 2 / y)
        ),
        3,
    )
    viss = [vis[:y, :x], vis[:y, x:], vis[y:, :x], vis[y:, x:]]
    if shows:
        boxx.shows(viss)
    return viss


def get_test_cams(feature_type="checkboard"):
    from calibrating import Cam, ArucoFeatureLib

    if feature_type == "checkboard":
        caml, camr, camd = Cam.get_test_cams()
        return dict(caml=caml, camr=camr, camd=camd)
    elif feature_type == "aruco":
        root = os.path.abspath(
            os.path.join(
                __file__,
                "../../../calibrating_example_data/paired_stereo_and_depth_cams_aruco",
            )
        )
        feature_lib = ArucoFeatureLib(occlusion=True)
        caml = Cam(
            glob(os.path.join(root, "*", "stereo_l.jpg")),
            feature_lib,
            name="caml",
            enable_cache=True,
        )
        camr = Cam(
            glob(os.path.join(root, "*", "stereo_r.jpg")),
            feature_lib,
            name="camr",
            enable_cache=True,
        )
        camd = Cam(
            glob(os.path.join(root, "*", "depth_cam_color.jpg")),
            feature_lib,
            name="camd",
            undistorted=True,
            enable_cache=True,
        )
        return dict(caml=caml, camr=camr, camd=camd)


class CvShow:
    """
    Object-oriented Pythonic cv2.imshow

    >>> with cvshow:
            for key in cvshow:
                cvshow.imshow(rgb, "Windows name")
                if key == "q":
                    break
    
    Or a more sample but not destroyAllWindows() when meet Exception 
    >>> for key in cvshow:
            cvshow.imshow(rgb, "Windows name")
            if key == "q":
                cvshow.breakk()
    """

    destroyed = False
    destroyed_tag_for_breakk = False
    temp_keys = []

    def __init__(self):
        pass

    def __enter__(self):
        CvShow.destroyed = False
        self.get_key()
        return self

    def __exit__(self, *args):
        self.destroyed = True
        cv2.destroyAllWindows()

    @classmethod
    def imshow(cls, rgb, window="default"):
        rgb = rgb[..., ::-1] if rgb.ndim == 3 and rgb.shape[-1] == 3 else rgb
        cv2.imshow(window, rgb)
        key_idx = cv2.waitKey(1)
        if key_idx != -1:
            cls.temp_keys.append(key_idx)

    @classmethod
    def get_key(cls):
        key_idx = cv2.waitKey(1)
        if key_idx != -1:
            cls.temp_keys.append(key_idx)
        if len(cls.temp_keys):
            key_idx = cls.temp_keys.pop(0)
        if 0 < key_idx and key_idx < 256:
            return chr(key_idx)
        return key_idx

    def __next__(self):
        if self.destroyed_tag_for_breakk:
            self.destroyed_tag_for_breakk = False
            raise StopIteration()
        return self.get_key()

    def __iter__(self):
        return self

    def __call__(self, rgb, window="default"):
        self.imshow(rgb, window=window)

    def breakk(self):
        self.__exit__()
        self.destroyed_tag_for_breakk = True

    end = breakk
    stop = breakk

    @classmethod
    def test(cls):
        import skimage.data

        rgb = skimage.data.coffee()
        with cvshow:
            for key in cvshow:
                rgb = np.append(rgb[1:], rgb[:1], 0)
                cvshow.imshow(rgb, "Scroll RGB image vertically by with")
                if key == "q":
                    break
        for key in cvshow:
            rgb = np.append(rgb[1:], rgb[:1], 0)
            cvshow.imshow(rgb, "Scroll RGB image vertically no with")
            if key == "q":
                cvshow.breakk()


cvshow = CvShow()


if __name__ == "__main__":
    from boxx import *

    CvShow.test()
