#!/usr/bin/env python3

import os
import cv2
import boxx
import numpy as np
from glob import glob


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
    return xyzs_to_arr2d(xyzs, xy[::-1])


def xyzs_to_arr2d(xyzs, hw=None, bg_value=0, arr2d=None):
    if arr2d is None:
        if hw is None:
            hw = np.int32(xyzs.max(0)[:2].round()) + 1
        arr2d = np.ones(hw, xyzs.dtype) * bg_value
    else:
        hw = arr2d.shape[:2]

    xs, ys = np.int32(xyzs[:, :2].round()).T
    mask = (xs >= 0) & (xs < hw[1]) & (ys >= 0) & (ys < hw[0])
    arr2d[ys[mask], xs[mask]] = xyzs[:, 2][mask]
    return arr2d


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


def vis_T(T, cam=None, img=None, length=0.1):
    if cam is None:
        from calibrating import Cam

        cam = Cam.get_example_720p()
    if img is None:
        vis = np.ones(list(cam.xy[::-1]) + [3], np.uint8) * 128
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
        feature_lib = ArucoFeatureLib()
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


if __name__ == "__main__":
    pass
