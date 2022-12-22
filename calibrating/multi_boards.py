#!/usr/bin/env python3

import boxx
import numpy as np
from collections import defaultdict
from functools import wraps

with boxx.inpkg():
    from .camera import Cam
    from .utils import mean_Ts
    from .reconstruction import convert_cam_to_nerf_json


class MultiBoardsCam(Cam):
    """
    Support multi calibration boards in one image
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
            cam_ = Cam(imgps, board=board, **kwargs)
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
        sorted(boxx.glob(f"{img_dir}/*left.jpg")),
        boards,
        name="boards_cam",
    )

    # Test convert boards_cam to NeRF.json
    boards_cam.convert_to_nerf_json(f"{img_dir}/instant-ngp.json")
