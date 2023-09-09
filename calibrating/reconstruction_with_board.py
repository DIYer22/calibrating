import cv2
import boxx
import numpy as np

with boxx.inpkg():
    from . import utils


def get_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    return sharpness


def convert_cam_to_nerf_json(
    cam,
    json_path=None,
    target_scale=None,  # 1~5m, recommend 2m
    aabb_scale=4,
    sharpness=True,
    basename=False,
    rotate_z_as_up=True,
    coordinate_type="nerf",
):
    assert coordinate_type in ("opencv", "nerf")
    nerf_json = dict(
        fl_x=cam.fx,
        fl_y=cam.fy,
        cx=cam.cx,
        cy=cam.cy,
        w=cam.xy[0],
        h=cam.xy[1],
        camera_angle_x=cam.fovs["fovx"] * np.pi / 180,
        camera_angle_y=cam.fovs["fovy"] * np.pi / 180,
        coordinate_type=coordinate_type,
    )
    distort = dict(zip("k1 k2 p1 p2".split(), cam.D[0, :4]))
    nerf_json.update(distort)
    nerf_json["frames"] = []
    for key, d in cam.items():
        if "T" not in d:
            continue
        if "T_cam_in_world" in d:
            T_cam_in_world = d["T_cam_in_world"]
        else:
            T_cam_in_world = np.linalg.inv(d["T"])

        if coordinate_type == "nerf":
            # Refrence https://github.com/NVlabs/instant-ngp/blob/25dec33c253f035485bb2e1f8563e12ef3134e8b/scripts/colmap2nerf.py#L287
            cv_to_nerf = np.array(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
            )
            T_cam_in_world = T_cam_in_world @ cv_to_nerf
            if rotate_z_as_up:
                T_cam_in_world = (
                    utils.r_t_to_T(
                        [np.pi, 0, 0],
                    )
                    @ T_cam_in_world
                )
        dic = dict(
            file_path=boxx.basename(d["path"]) if basename else d["path"],
            transform_matrix=T_cam_in_world,
        )
        if sharpness:
            img = boxx.imread(d["path"])
            dic["sharpness"] = get_sharpness(img)
        nerf_json["frames"].append(dic)
    if target_scale is not None:
        mean_t = np.mean(
            [np.linalg.norm(d["transform_matrix"][:3, 3]) for d in nerf_json["frames"]]
        )
        for d in nerf_json["frames"]:
            d["transform_matrix"][:3, 3] *= target_scale / mean_t
    if json_path is None:
        return nerf_json
    return boxx.savejson(nerf_json, json_path, indent=1)


if __name__ == "__main__":
    import os
    from calibrating import CharucoBoard, Cam
    from boxx import *

    example_data_dir = os.path.abspath(
        os.path.join(
            __file__,
            "../../../calibrating_example_data",
        )
    )
    recon_dir = os.path.join(example_data_dir, "reconstruction_with_marker_board")
    glob_path = recon_dir + "/*.jpg"

    board = CharucoBoard.build_with_calibration_img(
        ppi=218.35,
        hw=(2480, 3508),
        n=12,
        aruco_dict_tag="DICT_4X4_1000",
        using_marker_corner=True,
    )

    cam = Cam(
        sorted(boxx.glob(glob_path)),
        board,
    )
    jsp = recon_dir + "/transforms.json"
    convert_cam_to_nerf_json(cam, json_path=jsp, basename=True, target_scale=2.5)
    print("Save to:", jsp)
