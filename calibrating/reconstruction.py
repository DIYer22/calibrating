import cv2
import boxx
import numpy as np

with boxx.inpkg():
    from . import utils


def convert_cam_to_instant_ngp_json(
    cam,
    json_path=None,
    target_scale=None,
    aabb_scale=4,
    sharpness=True,
    basename=False,
    rotate_z_as_up=True,
):
    def get_sharpness(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        return sharpness

    nerf_json = dict(
        fl_x=cam.fx,
        fl_y=cam.fy,
        cx=cam.cx,
        cy=cam.cy,
        w=cam.xy[0],
        h=cam.xy[1],
        camera_angle_x=cam.fovs["fovx"] * np.pi / 180,
        camera_angle_y=cam.fovs["fovy"] * np.pi / 180,
    )
    distort = dict(zip("k1 k2 p1 p2".split(), cam.D[0, :4]))
    nerf_json.update(distort)
    nerf_json["frames"] = []
    for key, d in cam.items():
        # Refrence https://github.com/NVlabs/instant-ngp/blob/25dec33c253f035485bb2e1f8563e12ef3134e8b/scripts/colmap2nerf.py#L287
        T_cam_to_word = np.linalg.inv(d["T"])
        cv_to_nerf = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        T_nerf = T_cam_to_word @ cv_to_nerf
        if rotate_z_as_up:
            T_nerf = utils.r_t_to_T([np.pi, 0, 0],) @ T_nerf
        dic = dict(
            file_path=boxx.basename(d["path"]) if basename else d["path"],
            transform_matrix=T_nerf,
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
    from calibrating import CharucoFeatureLib, Cam
    from boxx import *

    example_data_dir = os.path.abspath(
        os.path.join(__file__, "../../../calibrating_example_data",)
    )
    recon_dir = os.path.join(example_data_dir, "reconstruction_with_marker_board")
    glob_path = recon_dir + "/*.jpg"

    feature_lib = CharucoFeatureLib.build_with_calibration_img(
        ppi=218.35,
        hw=(2480, 3508),
        n=12,
        aruco_dict_tag="DICT_4X4_1000",
        using_marker_corner=True,
    )

    cam = Cam(sorted(boxx.glob(glob_path)), feature_lib,)
    jsp = recon_dir + "/transforms.json"
    convert_cam_to_instant_ngp_json(cam, json_path=jsp, basename=True, target_scale=2.5)
    print("Save to:", jsp)
