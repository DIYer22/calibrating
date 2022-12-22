#!/usr/bin/env python3

import os
import cv2
import boxx
import numpy as np
from glob import glob
from boxx import imread, imsave, shows, show, showb, tree, loga, pi

inv = np.linalg.inv


def R_t_to_T(R, t=None):
    if t is None:
        t = np.zeros((3,))
    T = np.zeros((4, 4))
    T[:3, :3] = R
    T[:3, -1] = np.array(t).squeeze()
    T[3, 3] = 1
    return T


def r_t_to_T(r, t=None):
    r = np.float32(r)
    assert r.size == 3
    return R_t_to_T(cv2.Rodrigues(r.squeeze())[0], t)


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


def project_vec_on_plane(v, plane_v):
    return v - np.dot(v, plane_v) / (np.linalg.norm(plane_v) ** 2) * plane_v


def rotate_shortest_of_two_vecs(v1, v2, return_rodrigues=False):
    cross = np.cross(v1, v2)
    rad = np.arccos((v1 * v2).sum() / np.linalg.norm(v1) / np.linalg.norm(v2))
    r = rad * cross / np.linalg.norm(cross)
    if return_rodrigues:
        return r
    return cv2.Rodrigues(r)[0]


def apply_T_to_point_cloud(T, point_cloud):
    """
    point_cloud's shape is N * [x, y, z] or N * [x, y, z, ....]
    """
    point_cloud_n4 = np.ones((len(point_cloud), 4))
    point_cloud_n4[:, :3] = point_cloud[:, :3]
    new_point_cloud = (T @ point_cloud_n4.T).T[:, :3]
    if point_cloud.shape[1] > 3:
        new_point_cloud = np.concatenate((new_point_cloud, point_cloud[:, 3:]), -1)
    return new_point_cloud


def rotate_depth_by_point_cloud(K, R, depth, interpolation_rate=1):
    pc = depth_to_point_cloud(depth, K, interpolation_rate=interpolation_rate)
    T = np.eye(4)
    T[:3, :3] = R
    new_pc = apply_T_to_point_cloud(T, pc)
    new_depth = point_cloud_to_depth(new_pc, K, depth.shape[::-1])
    return dict(depth=new_depth, R=R)


def rotate_depth_by_remap(K, R, depth, maps=None, xy_target=None, K_target=None):
    y, x = depth.shape
    if K_target is None:
        K_target = K
    if xy_target is None:
        xy_target = x, y
    else:
        # With the optical center as the midpoint, expand image xy
        K_target = K_target.copy()
        # K_target[:2, 2] += (np.array(xy_target) - (x, y))/2
    if maps is None:
        maps = cv2.initUndistortRectifyMap(
            K,
            None,
            R,
            K_target,
            xy_target,
            cv2.CV_32FC1,
        )
    ys, xs = np.mgrid[:y, :x]
    ys, xs = ys.flatten(), xs.flatten()
    points = np.array([xs, ys, np.ones_like(xs)]) * depth.flatten()[None]
    # points.shape is (3, n)
    new_points = (R @ np.linalg.inv(K) @ points).T
    new_depth_on_old_xy = new_points[:, 2].reshape(y, x)

    new_depth = cv2.remap(new_depth_on_old_xy, maps[0], maps[1], cv2.INTER_NEAREST)
    return dict(depth=new_depth, R=R, maps=maps)


def _get_appropriate_interpolation_rate(cam1, cam2, interpolation=1.5):
    if interpolation:
        interpolation_rate = cam1.K[0, 0] / cam2.K[0, 0] * interpolation
        if interpolation >= 1:
            interpolation_rate = max(interpolation_rate, 1)
    else:
        interpolation_rate = 1
    return interpolation_rate


def depth_to_point_cloud(depth, K, interpolation_rate=1, return_xyzuv=False):
    # depth to point cloud
    # K@Ps/z=xy
    assert depth.ndim == 2
    y, x = depth.shape
    if depth.dtype == np.uint16:
        depth = np.float32(depth / 1000.0)
    if interpolation_rate == 1:
        mask = depth != 0
        vs, us = np.mgrid[
            :y,
            :x,
        ][:, mask]

        # points = (np.array([xs, ys, np.ones_like(xs)]) * depth[None])[:, depth != 0].T
        points = (np.array([us, vs, np.ones_like(us)]) * depth[mask]).T

    else:
        y_, x_ = int(round(y * interpolation_rate)), int(round(x * interpolation_rate))
        depth_ = cv2.resize(depth, (x_, y_), interpolation=cv2.INTER_NEAREST)
        mask = depth_ != 0
        vs, us = (
            np.mgrid[
                :y_,
                :x_,
            ][:, mask]
            / interpolation_rate
        )
        points = (np.array([us, vs, np.ones_like(us)]) * depth_[mask]).T

    point_cloud = (np.linalg.inv(K) @ points.T).T
    if return_xyzuv:
        xyzuvs = np.concatenate([point_cloud, us[:, None], vs[:, None]], -1)
        return xyzuvs
    return point_cloud


def point_cloud_to_depth(points, K, xy):
    return point_cloud_to_arr2d(points, K, xy)


def point_cloud_to_arr2d(points, K, xy, values=None, bg_value=0):
    """
    Converts 3D point cloud data into a 2D image array.

    Parameters
    ----------
    points : ndarray
        A Nx3 array of 3D points.
    K : ndarray
        A 3x3 matrix used to perform a projection transformation on the points.
    xy : tuple of int
        A tuple of the form (width, height) specifying the dimensions of the
        desired 2D image array.
    values : ndarray, optional
        A 2D array of shape (N, C) representing data associated with each 3D point. Defaults to None.
    bg_value : int, optional
        The background value of the generated 2D image array.

    Returns
    -------
    arr2d : ndarray
        A 2D image array of shape (y, x, C) representing the converted 3D point cloud.
    """
    # equl to xyzs = (K @ points.T).T
    xyzs = points @ K.T
    xyzs[:, :2] /= xyzs[:, 2:]
    sorted_idx = np.argsort(-xyzs[:, 2])
    xyzs_sorted = xyzs[sorted_idx]
    if values is None:
        return uvzs_to_arr2d(xyzs_sorted, hw=xy[::-1], bg_value=bg_value)
    else:
        values_sorted = values[sorted_idx]
        return uvzs_to_arr2d(
            xyzs_sorted, hw=xy[::-1], values=values_sorted, bg_value=bg_value
        )


def uvzs_to_arr2d(uvs, hw=None, bg_value=0, arr2d=None, values=None):
    """
    uvs: numpy array with shape (n, 2) or (n, 2 + values.shape[1]), where n is the number of (u, v) coordinates
    hw: tuple of ints, representing the height and width of the resulting array. If not specified, it will default to the maximum (u, v) coordinates rounded up.
    bg_value: int or float, representing the background value to fill the resulting array with.
    arr2d: default arr2d. If not specified, a new array will be created with the specified height and width.
    values: numpy array with shape (n, values.shape[1]), where values.shape[1] is the number of values to be associated with each (u, v) coordinate.
    """
    if values is None:
        uvs, values = uvs[:, :2], uvs[:, 2:]
    if values.ndim == 1:
        values = values[:, None]
    if arr2d is None:
        if hw is None:
            hw = np.int32(uvs.max(0)[:2].round()) + 1
        if values.shape[1] >= 2:
            hw = tuple(hw) + (values.shape[1],)
        arr2d = np.ones(hw, values.dtype) * bg_value
    else:
        hw = arr2d.shape[:2]

    xs, ys = np.int32(uvs[:, :2].round()).T
    mask = (xs >= 0) & (xs < hw[1]) & (ys >= 0) & (ys < hw[0])
    arr2d[ys[mask], xs[mask]] = (
        values[mask] if values.shape[1] >= 2 else values[mask][:, 0]
    )
    return arr2d


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


def get_reproject_remap(K1, K2, T_2in1, depth2, xy1, interpolation_rate=1):
    """
    get reproject remap of cam2 to cam1
    """
    xyzuvs2 = depth_to_point_cloud(
        depth2, K2, interpolation_rate=interpolation_rate, return_xyzuv=True
    )
    xyzs2, uvs = xyzuvs2[:, :3], np.float32(xyzuvs2[:, 3:])
    xyzs1 = apply_T_to_point_cloud(T_2in1, xyzs2)
    reproject_remap = point_cloud_to_arr2d(
        xyzs1, K1, xy=xy1, values=uvs, bg_value=-1
    ).transpose(2, 0, 1)
    return reproject_remap


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
        output = spline(
            input_uvs[:, 0],
            input_uvs[:, 1],
        )
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


def vis_depth_l1(re, gt=0, max_l1=None, overexposed=True):
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
        max_l1 to vis. area of abs(l1) > max_l1 will overexposed
        if max_l1 < 0 and in (-1 ~ 0), the max_l1 = top k% of l1
        The default None is -0.05 or top 5% as overexposed.
    """
    GREY_CLIP = 0.1  # how_grey_to_distinguish missing depth and l1==0
    if isinstance(gt, (int, float, np.number)):
        gt = np.ones_like(re) * gt

    mask_valid = np.bool8(re) & np.bool8(gt)
    l1 = (re - gt) * mask_valid
    abs_l1 = np.abs(l1)
    if max_l1 is None:
        max_l1 = -0.05 if overexposed else 0
    if max_l1 == 0 or max_l1 <= -1:
        max_l1 = abs_l1.max()
    elif max_l1 < 0:
        valid_num = mask_valid.sum()
        if valid_num:
            k = int(-max_l1 * valid_num)
            k = max(k, 0)
            overexposed_l1 = -np.partition(-abs_l1[mask_valid], k)[: k + 1]
            max_l1 = overexposed_l1.min()
        else:
            max_l1 = 1.0
    l1_gt_0 = l1 > 0
    l1_lt_0 = l1 < 0
    # l1>0(far) red, l1<0(near) green
    l1_vis_ = np.array([l1 * (l1_gt_0), -l1 * (l1_lt_0), l1 * 0])
    l1_vis_norma = l1_vis_.clip(0, max_l1) / max_l1
    l1_vis_with_grey = (l1_vis_norma * (1 - GREY_CLIP) + GREY_CLIP) * mask_valid
    l1_vis = np.uint8(l1_vis_with_grey * 255).transpose(1, 2, 0)
    if overexposed:
        overexposed_mask = abs_l1 > max_l1
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
    if hasattr(color, "tolist"):
        color = tuple(color.tolist())
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
        vis = cv2.circle(
            vis,
            tuple(uv[:2]),
            size,
            _color,
            -1,
        )
    return vis


def _get_vis_background_of_cam(cam):
    # New idea different size boxes in 1m to project on cam's image with K
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
    invert_color = T[2, 3] < 0
    if invert_color:
        vis = 255 - vis
    cv2.drawFrameAxes(vis, cam.K, cam.D, rvec, tvec, length)
    if invert_color:
        vis = 255 - vis
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
    vis = np.concatenate(
        (
            visv,
            vis_stereo(img2, img1, n_line=n_line),
        ),
        0,
    )
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
    from calibrating import Cam, PredifinedArucoBoard1

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
        board = PredifinedArucoBoard1()
        caml = Cam(
            glob(os.path.join(root, "*", "stereo_l.jpg")),
            board,
            name="caml",
            enable_cache=True,
        )
        camr = Cam(
            glob(os.path.join(root, "*", "stereo_r.jpg")),
            board,
            name="camr",
            enable_cache=True,
        )
        camd = Cam(
            glob(os.path.join(root, "*", "depth_cam_color.jpg")),
            board,
            name="camd",
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
