#!/usr/bin/env python3

import cv2
import boxx
import numpy as np
import random


def vis_flow(flow, img=None, arrowe=50):
    _, h, w = flow.shape
    hsv = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    hsv[..., 2] = 255
    mag, ang = cv2.cartToPolar(flow[0], flow[1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if img is not None:
        hsv[..., 1]
        rate = 0.5
        flow_vis = np.uint8(img * rate + flow_vis * (1 - rate))
    # vis arrowe
    if isinstance(arrowe, float):
        arrowe = int((h * w) * arrowe + 0.6)
    if arrowe:
        arrowe_vis = flow_vis.copy()
        # yxs_valid = np.mgrid[:h, :w][:, mask].T
        # yxs_sampled = random.sample(list(yxs_valid), int(arrowe))
        yxs = np.mgrid[:h, :w].reshape(2, -1).T
        yxs_sampled = random.choices(
            list(yxs), weights=np.abs(flow).sum(0).reshape(-1), k=int(arrowe)
        )
        for y, x in yxs_sampled:
            x_, y_ = np.int32(flow[:, y, x] * (w, h) + (x, y))
            color = flow_vis[y, x].tolist()

            cv2.arrowedLine(
                arrowe_vis,
                (x, y),
                (
                    x_,
                    y_,
                ),
                (128, 128, 128),
                2,
                tipLength=0.05,
            )
            cv2.arrowedLine(
                arrowe_vis,
                (x, y),
                (
                    x_,
                    y_,
                ),
                color,
                1,
                tipLength=0.05,
            )
            cv2.circle(
                arrowe_vis,
                (
                    x_,
                    y_,
                ),
                3,
                (128, 128, 128),
                -1,
            )
            cv2.circle(
                arrowe_vis,
                (
                    x_,
                    y_,
                ),
                2,
                color,
                -1,
            )
        flow_vis = arrowe_vis
    return flow_vis


def flow_abs_to_normal(flow_abs):
    h, w, _ = flow_abs.shape
    flow = flow_abs.transpose(2, 0, 1) / [[[w]], [[h]]]
    return np.float32(flow)


def flow_normal_to_abs(flow, hw=None):
    if hw is None:
        _, h, w = flow.shape
    else:
        h, w = hw
    flow_abs = (flow * [[[w]], [[h]]]).transpose(1, 2, 0)
    return flow_abs


def warp_flow(flow, img1=None, img2=None, interpolation=cv2.INTER_LINEAR):
    """通过 remap 来 warp flow, 生成新的图像.
    若输入 img1 则输出 img2_warped, 但存在多个像素对应一个像素, 导致稀疏孔洞
    若输入 img2 这输出 img1_warped, 没有稀疏孔洞效果好, 更推荐

    Args:
        flow (np.ndarray): flow_normal [-1~1] 2*h*w
        img1 (np.ndarray, optional): pre frame
        img2 (np.ndarray, optional): next frame
    Returns:
        warped image
    """
    _, h, w = flow.shape
    remap_xy = np.float32(np.mgrid[:h, :w][::-1])
    remap_flow = flow * [[[w]], [[h]]]
    if img1 is not None:
        uv_new = (remap_xy + remap_flow).round().astype(np.int32)
        mask = (
            flow.any(0)
            & (uv_new[0] >= 0)
            & (uv_new[1] >= 0)
            & (uv_new[0] < w)
            & (uv_new[1] < h)
        )
        uv_new_ = uv_new[:, mask]
        remap_xy[:, uv_new_[1], uv_new_[0]] = remap_xy[:, mask]
        remap_x, remap_y = remap_xy
        return cv2.remap(img1, remap_x, remap_y, interpolation)
    elif img2 is not None:
        remap_x, remap_y = np.float32(remap_xy + remap_flow)
        img2 = boxx.resize(img2, (h, w))
        return cv2.remap(img2, remap_x, remap_y, interpolation)


if __name__ == "__main__":
    from boxx import *
