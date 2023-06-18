#!/usr/bin/env python3
import os
import cv2
import tqdm
import boxx
import glob
import pickle
import numpy as np
import calibrating
from calibrating import vis_point_uvs, apply_T_to_point_cloud, vis_depth


def xyzs_to_dense_depth(xyzs, cam, T=None, rate=0.5, sparse=False):
    if T is not None:
        xyzs = apply_T_to_point_cloud(T, xyzs)
    w, h = cam.xy
    xyzs_ = xyzs @ cam.K.T
    xyzs_ = xyzs_[xyzs_[:, 2] > 0]
    uvs = xyzs_[:, :2] / xyzs_[:, 2:3]
    idx = (uvs.min(-1) >= 0) & (uvs[:, 0] < w) & (uvs[:, 1] < h)
    uvs = uvs[idx]
    uvzs = np.concatenate((uvs, xyzs[idx, 2:3]), -1)
    if rate is None:
        if len(xyzs) > 5000:
            rate = (len(xyzs) / h / w) ** 0.5
        else:
            rate = 0.25
    if sparse:
        depth = calibrating.uvzs_to_arr2d(
            uvzs * [[rate, rate, 1]],
            [int(round(h * rate)), int(round(w * rate))],
        )
    else:
        depth = calibrating.interpolate_uvzs(
            uvzs * [[rate, rate, 1]],
            [int(round(h * rate)), int(round(w * rate))],
            constrained_type="convex_hull",
            inter_type="nearest",
        )
    return boxx.resize(depth, [h, w])


if __name__ == "__main__":

    img_dir = "calibrate_lidar2cam_by_PnP_example_data"
    if not os.path.exists(img_dir):
        cmd = "git clone https://github.com/yl-data/calibrate_lidar2cam_by_PnP_example_data"
        print("Not exists example_data, will automatic clone it by CMD:", cmd)
        os.system(cmd)

    cam = calibrating.Cam.load(
        """
name: cam_intrinsic
D:
- - -0.22013946553626615
  - 0.07699891879724093
  - -0.0002214795521067203
  - -0.00021607328691930825
  - -0.013607369975999666
_calibrating_version: 0.6.7
cx: 821.5502789828007
cy: 468.6433131586491
fx: 803.5323604817661
fy: 803.8263591371266
retval: 0.18045235630347495
xy:
- 1624
- 900
"""
    )

    cam_pc2depth = cam.copy()
    cam_pc2depth.D[:] = 0
    for xyzs_path in tqdm.tqdm(sorted(glob.glob(img_dir + "/*xyzs.pkl"))):
        depth_vis_path = xyzs_path.replace("xyzs.pkl", "depth.png")
        if os.path.exists(depth_vis_path):
            continue
        xyzs = pickle.load(open(xyzs_path, "rb"))
        depth = xyzs_to_dense_depth(xyzs, cam_pc2depth)

        pickle.dump(depth, open(xyzs_path.replace("xyzs.pkl", "depth.pkl"), "wb"))
        depth_vis = vis_depth(depth)
        boxx.imsave(depth_vis_path, depth_vis)
        # tree-xyzs

    # Annotate some matched points in both camera image and visualization image of depth
    # The number of matched points should >= 5 for PnP algorithm
    # It is necessary to cover corners
    # The distance to the marked points should be large (>5m) to avoid depth errors in LiDAR

    # matched_uvs_str format: img_file_name, uv_on_img, uv_on_depth
    matched_uvs_str = """
0_img.png, 446 337, 431 346
1_img.png, 766 588, 780 604
2_img.png, 785 322, 793 348
3_img.png, 1318 314, 1382 316
4_img.png, 158 493, 58 514
5_img.png, 769 417, 788 430
6_img.png, 198 584, 111 620
7_img.png, 1222 552, 1265 572
8_img.png, 649 294, 656 306
9_img.png, 554 309, 556 321
    """

    uvs = []
    xyz_matcheds = []
    for line in matched_uvs_str.strip().split("\n"):
        name, uv_img, uv_depth = line.strip().split(",")
        uv_img, uv_depth = np.array(boxx.findints(uv_img)), np.array(
            boxx.findints(uv_depth)
        )
        img_path = os.path.join(img_dir, name)
        xyzs = pickle.load(open(img_path.replace("img.png", "xyzs.pkl"), "rb"))
        # depth = xyzs_to_dense_depth(xyzs, cam_pc2depth)
        depth = pickle.load(open(img_path.replace("img.png", "depth.pkl"), "rb"))

        z = depth[uv_depth[1], uv_depth[0]]
        uvz = np.pad(uv_depth, (0, 1), constant_values=1) * z
        xyz_matched = ([uvz] @ np.linalg.inv(cam_pc2depth.K).T)[0]
        uvs.append(uv_img)
        xyz_matcheds.append(xyz_matched)
        # if xyz_matched[-1]: continue
        img = boxx.imread(img_path)
        # img_vis = vis_point_uvs(uv_img, img,size=2.5)
        # depth_vis = vis_point_uvs(uv_depth, boxx.imread(img_path.replace("img.png", "depth.png")), size=2.5)
        # boxx.shows(img_vis, depth_vis)
        # 1/0
    uvs = np.float64(uvs)
    xyz_matcheds = np.float64(xyz_matcheds)
    # get cam's T by solvePnP
    retval, rvecs, tvecs, reprojectionError = cv2.solvePnPGeneric(
        xyz_matcheds, uvs[:, None], cam.K, cam.D
    )
    assert retval == 1
    print("reprojectionError:", reprojectionError)
    T_lidar_in_cam = calibrating.r_t_to_T(rvecs[0], tvecs[0])

    # %%
    depth = xyzs_to_dense_depth(xyzs, cam, T=T_lidar_in_cam, rate=1, sparse=1)
    # depth[depth>20] = 0
    depth[depth < 3] = 0

    vis = vis_point_uvs(uvs, img, size=4.9, color=(255, 255, 0))
    vis_undistort = cv2.undistort(vis, cam.K, cam.D)
    uv_matcheds = apply_T_to_point_cloud(T_lidar_in_cam, xyz_matcheds) @ cam.K.T
    uv_matcheds = uv_matcheds[:, :2] / uv_matcheds[:, 2:]
    vis_undistort = vis_point_uvs(uv_matcheds, vis_undistort, size=2.1)
    boxx.shows(vis_depth(depth), vis_undistort, png=True)
