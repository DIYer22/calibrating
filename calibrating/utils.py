#!/usr/bin/env python3

import boxx
from boxx import *
from boxx import npa
import cv2
import numpy as np


def R_t_to_T(R, t):
    T = np.zeros((4, 4))
    T[:3, :3] = R
    T[:3, -1] = t
    T[3, 3] = 1
    return T


def r_t_to_T(r, t):
    return R_t_to_T(cv2.Rodrigues(r.squeeze())[0], t.squeeze())


def T_to_r(T):
    r = cv2.Rodrigues(T[:3, :3])[0]
    return r


def apply_T_to_point_cloud(T, point_cloud):
    new_point_cloud = np.ones((len(point_cloud), 4))
    new_point_cloud[:, :3] = point_cloud
    return (T @ new_point_cloud.T).T[:, :3]


def depth_to_point_cloud(depth, K, resize=1):
    # depth to point cloud
    # K@Ps/z=xy
    # TODO resize
    assert depth.ndim == 2
    ys, xs = np.mgrid[
        : depth.shape[0], : depth.shape[1],
    ]

    _points = (npa([xs, ys, np.ones_like(xs)]) * depth[None])[:, depth != 0].T
    point_cloud = (np.linalg.inv(K) @ _points.T).T
    return point_cloud


def point_cloud_to_depth(points, K, xy):
    xyzs = (K @ points.T).T
    xyzs[:, :2] /= xyzs[:, 2:]

    xyzs = npa - sorted(xyzs, key=lambda xyz: -xyz[2])
    mask = (
        (xyzs[:, 0] >= 0)
        & (xyzs[:, 0] < xy[0])
        & (xyzs[:, 1] >= 0)
        & (xyzs[:, 1] < xy[1])
    )
    xyzs = xyzs[mask, :]

    depth = np.zeros(xy[::-1])
    depth[np.int32(xyzs[:, 1].round(0)), np.int32(xyzs[:, 0].round(0))] = xyzs[:, 2]
    return depth
    # test point_cloud_to_depth
    depth2 = point_cloud_to_depth(points_in_m, intrinsicm["K"], intrinsicm["xy"])
    boxx.shows(depth, depth2, boxx.norma)
    assert (depth == depth2).all()


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


if __name__ == "__main__":
    pass
