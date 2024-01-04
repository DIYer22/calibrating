#!/usr/bin/env python3

import boxx
import numpy as np
from collections import defaultdict
from functools import wraps

with boxx.inpkg():
    from .camera import Cam
    from .utils import mean_Ts, apply_T_to_point_cloud
    from .reconstruction_with_board import convert_cam_to_nerf_json
    from .boards import BaseBoard


class MultiBoards(BaseBoard):
    def __init__(self, boards, imgs=None, cam=None):
        if isinstance(boards, dict):
            boards = [boards[k] for k in sorted(boards)]
        self.boards = boards
        self.Ts_all = []
        if imgs is not None:
            self.add_board_imgs(imgs, cam)

    def add_board_imgs(self, imgs, cam=None):
        boards = self.boards
        if isinstance(imgs, np.ndarray) and imgs.ndim <= 3:
            imgs = [imgs]
        if cam is None:
            cam = self.cam
        else:
            self.cam = cam

        for img in imgs:
            # TODO No need all board valid
            Ts_in_cam = [
                cam.get_calibration_board_T(img, board)["T"] for board in boards
            ]
            T_cam_in_board0 = np.linalg.inv(Ts_in_cam[0])
            Ts = [T_cam_in_board0 @ T for T in Ts_in_cam]
            self.Ts_all.append(Ts)

        new_obj_points = {}
        self.Ts = [
            mean_Ts([Ts[boardi] for Ts in self.Ts_all]) for boardi in range(len(boards))
        ]
        for boardi in range(len(boards)):
            T = self.Ts[boardi]
            object_points = boards[boardi].object_points
            if not isinstance(object_points, dict):
                object_points = {None: object_points}
            for k, object_point in object_points.items():
                new_obj_points[(boardi, k)] = apply_T_to_point_cloud(T, object_point)
        self.object_points = self.set_origin_to_center(new_obj_points)

    def find_image_points(self, d):
        boards = self.boards
        img = d["img"]
        all_image_points = {}
        for boardi in range(len(boards)):
            board = boards[boardi]
            d_ = dict(img=img)
            board.find_image_points(d_)
            image_points = d_.get("image_points")
            if image_points is None:
                continue
            if not isinstance(image_points, dict):
                image_points = {None: image_points}
            for k, image_point in image_points.items():
                all_image_points[(boardi, k)] = image_point
        if len(all_image_points) > 0:
            object_points = {id: self.object_points[id] for id in all_image_points}
            d.update(
                image_points=all_image_points,
                object_points=object_points,
            )


class MultiBoardsCam(Cam):
    """
    Cam class that support multi calibration boards in one image
    Each board dict's key is named with `{img_key}~{board_key}`
    """

    def __init__(self, imgps=None, boards=None, **kwargs):
        """
        boards: dict or list of board
        """
        super().__init__(**kwargs)
        if imgps is None:
            return
        if not isinstance(boards, dict):
            boards = dict([(str(i), v) for i, v in enumerate(boards)])
        kwargs["save_feature_vis"] = False
        for name in boards:
            kwargs["name"] = self.name + "~cam_~" + name
            board = boards[name]
            try:
                cam_ = Cam(imgps, board=board, **kwargs)
            except AssertionError:
                continue
            for k in cam_:
                cam_[k]["board"] = board
                self[k + "~" + name] = cam_[k]
        self.calibrate()
        self.boards = boards

    def get_board_name_to_T_world(self, origin_to_center=False):
        # TODO 1. support no common view, 2. using 加权平均(Rotate 向量模长为距离) or 优化重投影 loss 来求 T
        boards = self.boards
        board_name_to_img_names = defaultdict(lambda: {})
        for key, d in self.items():
            img_name, board_name = key.split("~")
            board_name_to_img_names[board_name][img_name] = d

        board_name0 = list(boards)[0]

        board_name_to_T_world = board_name_to_T_board_in_board0 = {}

        for board_name in boards:
            ds0 = board_name_to_img_names[board_name0]
            ds = board_name_to_img_names[board_name]
            Ts = []
            for img_name in set(ds).intersection(ds0):
                if "T" in ds[img_name] and "T" in ds0[img_name]:
                    Ts.append(np.linalg.inv(ds0[img_name]["T"]) @ ds[img_name]["T"])
            assert Ts, f"{board_name} has common vision with first board: {board_name0}"
            board_name_to_T_board_in_board0[board_name] = mean_Ts(Ts)
        if origin_to_center:
            to_T_world = np.linalg.inv(
                mean_Ts(list(board_name_to_T_board_in_board0.values()))
            )
            board_name_to_T_world = {
                n: to_T_world @ T for n, T in board_name_to_T_board_in_board0.items()
            }
        return board_name_to_T_world

    @wraps(convert_cam_to_nerf_json)
    def convert_to_nerf_json(self, *args, **argkws):
        boards = self.boards
        board_name_to_T_world = self.get_board_name_to_T_world(True)
        board_name_to_img_names = defaultdict(lambda: {})
        for key, d in self.items():
            img_name, board_name = key.split("~")
            board_name_to_img_names[board_name][img_name] = d

        board_name0 = list(boards)[0]
        ds_new = board_name_to_img_names[board_name0].copy()
        img_names = set(sum([list(d) for d in board_name_to_img_names.values()], []))
        for img_name in img_names:
            T_cam_in_worlds = []
            for board_name, ds in board_name_to_img_names.items():
                if "T" in ds.get(img_name, ""):
                    T_cam_in_worlds.append(
                        board_name_to_T_world[board_name]
                        @ np.linalg.inv(ds[img_name]["T"])
                    )

            if T_cam_in_worlds:
                ds_new[img_name]["T_cam_in_world"] = mean_Ts(T_cam_in_worlds)

        cam_new = self.copy()
        cam_new.update(ds_new)
        convert_cam_to_nerf_json(cam_new, *args, **argkws)


if __name__ == "__main__":
    from boxx import *
    import os
    import calibrating
    from calibrating import Cam, A3, A4, CharucoBoard

    img_dir = os.path.abspath(
        os.path.join(
            __file__,
            "../../../calibrating_example_data/multi_boards_A3_5boards_4x4",
        )
    )
    imgps = sorted(boxx.glob(f"{img_dir}/*left.jpg"))
    # prepare boards
    n = 12
    marker_names = [f"DICT_4X4_1000_start{i*(n**2+1)//2}" for i in [0, 1, 2, 3, 4]]
    boards = {}
    for name in marker_names:
        board = CharucoBoard.build_with_calibration_img(
            ppi=218.35, hw=(2480, 3508), n=n, aruco_dict_tag=name
        )
        boards[name] = board
    print("boards:\n", boards)

    boards_cam = MultiBoardsCam(
        imgps,
        boards,
        name="boards_cam",
    )

    # Test convert boards_cam to NeRF's transforms.json
    jsp = f"{img_dir}/transforms.json"
    boards_cam.convert_to_nerf_json(jsp)
    print("Save to:", jsp)

    #%% Test MultiBoards as one board
    mboard = MultiBoards(boards, imgps[:1], boards_cam)
    boxx.shows - [mboard.vis_img(imgps[i], boards_cam) for i in range(4)]
