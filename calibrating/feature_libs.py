#!/usr/bin/env python3
import cv2
import boxx
import numpy as np
from collections import namedtuple

with boxx.inpkg():
    from . import utils

PredefPrinter = namedtuple("PredefPrinter", ["hw", "ppi"])
A4 = PredefPrinter((2480, 3508), 321.29)
A3 = PredefPrinter((3508, 4961), 308.86)
surface_book2_inch15 = PredefPrinter(hw=(2160, 3240), ppi=260)


class MetaFeatureLib:
    def find_image_points(self, d):
        """
        d is a dict for each checkboard image, including keys like "img", "path"
        
        Please calculate d["image_points"] and d["object_points"] base on d["img"]
    
        The method should:
            Set d["image_points"] as np.array of shape(n, 2) or {id: shape(n, 2)}
            Set d["object_points"] as np.array of shape(n, 3) or {id: shape(n, 3)}
            You could store some other important data in dict d
        """
        raise NotImplementedError()

    def vis(self, d, cam=None):
        """
        d is a dict for each checkboard image, including keys like "img", "path"
        
        return:
            vis: np.array(h, w, 3)
        """
        vis = d["img"].copy()
        if cam is not None and "T" in d:
            vis = utils.vis_T(d["T"], cam, vis)
        if len(d.get("image_points", "")):
            image_points = d["image_points"]
            if isinstance(image_points, dict):
                image_points = np.concatenate(list(image_points.values()), 0)
            vis = utils.vis_point_uvs(image_points, vis)
        return vis

    @classmethod
    def build_with_calibration_img(cls, **kwargs):
        # build init_kwargs and calibration_img code
        raise NotImplementedError(
            "Could go to https://calib.io/pages/camera-calibration-pattern-generator to genearte calibration image"
        )
        self = cls(**init_kwargs)
        self.init_kwargs = init_kwargs
        self.calibration_img = calibration_img
        self.calibration_img_info = info
        self.calibration_img_name = "xx mm, h*w.png"
        return self


class CheckboardFeatureLib(MetaFeatureLib):
    def __init__(self, checkboard=(11, 8), size_mm=25):
        self.checkboard = checkboard
        self.size_mm = size_mm
        self.object_points = np.zeros(
            (self.checkboard[0] * self.checkboard[1], 3), np.float32
        )
        self.object_points[:, :2] = np.float32(
            np.mgrid[: self.checkboard[0], : self.checkboard[1]].T.reshape(-1, 2)
            * self.size_mm
            / 1000
        )

    def find_image_points(self, d):
        img = d["img"]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.checkboard, None)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        if ret:
            cv2.cornerSubPix(
                gray, corners, (4, 4), (-1, -1), criteria,
            )
            d["corners"] = corners
            d["image_points"] = corners[:, 0]
            d["object_points"] = self.object_points

    def vis(self, d, cam=None):
        vis = d["img"].copy() if "img" in d else boxx.imread(d["path"])
        if cam is not None and "T" in d:
            vis = utils.vis_T(d["T"], cam, vis)
        h, w = vis.shape[:2]
        draw_subpix = h < 1500
        if draw_subpix:
            sub_size = 2
            vis = cv2.resize(vis, (w * sub_size, h * sub_size))
            cv2.drawChessboardCorners(
                vis, self.checkboard, d["corners"] * sub_size, True
            )
        else:
            cv2.drawChessboardCorners(vis, self.checkboard, d["corners"], True)
        return vis

    @classmethod
    def build_with_calibration_img(cls, hw=(1080, 1920), n=15, ppi=300):
        height, width = hw
        size = height // (n + 1)
        size_mm = round(size / ppi * 25.4, 2)

        hn = int((height - size) / size)
        wn = int((width - size) / size)

        # The length and width need to be parity with each other to prevent multiple legal external parameters caused by rotational symmetry
        if (wn % 2) == (hn % 2):
            wn -= 1

        img = np.ones((height, width), dtype=np.uint8) * 255

        for hidx in range(hn):
            for widx in range(wn):
                color = [0, 255][(hidx + widx) % 2]
                p = [hidx * size + size // 2, widx * size + size // 2]
                img[p[0] : p[0] + size, p[1] : p[1] + size] = color

        init_kwargs = dict(checkboard=(wn - 1, hn - 1), size_mm=25)
        self = cls(**init_kwargs)
        self.init_kwargs = init_kwargs
        self.calibration_img = img
        self.calibration_img_info = dict(hw=hw, ppi=ppi, **init_kwargs)
        self.calibration_img_name = f"checkboard_hw{hn-1}x{wn-1}_size{size_mm}mm.png"
        return self


def get_aruco_dictionary_with_start(aruco_dict_tag):
    """
    Parameters
    ----------
    aruco_dict_tag : int, str
        e.g. DICT_4X4_250_start50 will start 50 from DICT_4X4_250

    Returns
    -------
    aruco_dictionary
    """
    aruco_dict_tag_key = aruco_dict_tag_int = aruco_dict_tag
    start_idx = 0
    if isinstance(aruco_dict_tag, str):
        START_PREFIX = "_start"
        if START_PREFIX in aruco_dict_tag:
            split_idx = aruco_dict_tag.index(START_PREFIX)
            start_idx = int(aruco_dict_tag[split_idx + len(START_PREFIX) :])
            aruco_dict_tag_key = aruco_dict_tag[:split_idx]
        aruco_dict_tag_int = getattr(cv2.aruco, aruco_dict_tag_key)
    aruco_dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_tag_int)
    if start_idx:
        # aruco_dictionary.bytesList = np.append(aruco_dictionary.bytesList[start_idx:], aruco_dictionary.bytesList[:start_idx], 0)
        aruco_dictionary.bytesList = aruco_dictionary.bytesList[start_idx:]
    return aruco_dictionary


class ArucoFeatureLib(MetaFeatureLib):
    def __init__(self, occlusion=False, detector_parameters=None):
        import cv2.aruco

        self.occlusion = occlusion
        self.detector_parameters = detector_parameters or {}
        aruco_temp_str = """480 240 0 580 240 0 580 340 0 480 340 0 480 120 0 580 120 0 580 220 0 480 220 0 480 0 0 580 0 0 
        580 100 0 480 100 0 0 360 0 100 360 0 100 460 0 0 460 0 480 360 0 580 360 0 580 460 0 480 460 0 480 480 0 580 480 0 
        580 580 0 480 580 0 360 480 0 460 480 0 460 580 0 360 580 0 240 480 0 340 480 0 340 580 0 240 580 0 120 480 0 220 
        480 0 220 580 0 120 580 0 0 480 0 100 480 0 100 580 0 0 580 0 0 240 0 100 240 0 100 340 0 0 340 0 0 120 0 100 120 0 
        100 220 0 0 220 0 0 0 0 100 0 0 100 100 0 0 100 0 120 0 0 220 0 0 220 100 0 120 100 0 240 0 0 340 0 0 340 100 0 240 
        100 0 360 0 0 460 0 0 460 100 0 360 100 0 120 120 0 220 120 0 220 220 0 120 220 0 240 120 0 340 120 0 340 220 0 240 
        220 0 360 120 0 460 120 0 460 220 0 360 220 0 120 360 0 220 360 0 220 460 0 120 460 0 360 360 0 460 360 0 460 460 0 
        360 460 0 240 360 0 340 360 0 340 460 0 240 460 0 120 240 0 220 240 0 220 340 0 120 340 0 360 240 0 460 240 0 460 340 
        0 360 340 0"""
        self.all_object_points = np.float32(
            np.array(boxx.findints(aruco_temp_str)).reshape(-1, 3) / 1000.0
        )
        self.object_points = dict(enumerate(self.all_object_points.reshape(-1, 4, 3)))
        self.aruco_dict_tag = cv2.aruco.DICT_6X6_250

    def find_image_points(self, d):
        img = d["img"]
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        parameters = cv2.aruco.DetectorParameters_create()
        self.detector_parameters.setdefault("polygonalApproxAccuracyRate", 0.008)
        [
            setattr(parameters, key, value)
            for key, value in self.detector_parameters.items()
        ]

        d["corners"], d["ids"], rejectedImgPoints = cv2.aruco.detectMarkers(
            gray, cv2.aruco.Dictionary_get(self.aruco_dict_tag), parameters=parameters
        )
        d["valid"] = d["ids"] is not None and (
            len(d["ids"]) == len(self.object_points) or self.occlusion
        )
        if d["valid"]:
            d["ids"] = d["ids"][:, 0] if d["ids"].ndim == 2 else d["ids"]
            d["image_points"] = dict(
                zip(d["ids"], [corner.squeeze() for corner in d["corners"]])
            )
            d["object_points"] = {id: self.object_points[id] for id in d["ids"]}

    def vis(self, d, cam=None):
        img = d["img"].copy() if "img" in d else boxx.imread(d["path"])
        cv2.aruco.drawDetectedMarkers(img, d["corners"], d["ids"])
        if cam is not None and len(d["corners"]):
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                d["corners"], 0.1, cam.K, cam.D
            )
            for i in range(rvec.shape[0]):
                cv2.aruco.drawAxis(img, cam.K, cam.D, rvec[i], tvec[i], 0.05)
        return img


class CharucoFeatureLib(MetaFeatureLib):
    def __init__(
        self,
        square_xy=(9, 5),
        square_size_mm=44.95,
        marker_size_mm=22.475,
        aruco_dict_tag=None,
    ):
        self.square_xy = square_xy
        if aruco_dict_tag is None:
            aruco_dict_tag = cv2.aruco.DICT_4X4_250

        self.aruco_dict_tag = aruco_dict_tag
        self.aruco_dictionary = get_aruco_dictionary_with_start(aruco_dict_tag)
        self.board = cv2.aruco.CharucoBoard_create(
            square_xy[0],
            square_xy[1],
            square_size_mm / 1000.0,
            marker_size_mm / 1000.0,
            self.aruco_dictionary,
        )

    def find_image_points(self, d):
        img = d["img"]
        marker_corners, marker_ids, rejected_img_points = cv2.aruco.detectMarkers(
            img, self.aruco_dictionary
        )

        d["marker_ids"] = marker_ids
        d["marker_corners"] = marker_corners
        if len(marker_corners) > 0:
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, img, self.board
            )
            if charuco_ids is not None:
                d["ids"] = charuco_ids[:, 0]
                d["image_points"] = dict(zip(d["ids"], charuco_corners))
                d["object_points"] = {
                    id: self.board.chessboardCorners[id][None] for id in d["ids"]
                }
                d["corners"] = charuco_corners

    def vis(self, d, cam=None):
        vis = super().vis(d, cam)
        if d.get("ids") is not None:
            cv2.aruco.drawDetectedCornersCharuco(vis, d["corners"], d["ids"])
            cv2.aruco.drawDetectedMarkers(vis, d["marker_corners"], d["marker_ids"])
        return vis

    @classmethod
    def build_with_calibration_img(
        cls, hw=A4.hw, n=10, ppi=A4.ppi, aruco_dict_tag=None
    ):
        height, width = hw
        size = max(hw) // (n)
        size_mm = round(size / ppi * 25.4, 2)

        hn = int((height) / size)
        wn = int((width) / size)

        square_xy = (wn, hn)
        init_kwargs = dict(
            square_xy=square_xy,
            square_size_mm=size_mm,
            marker_size_mm=size_mm * 0.75,
            aruco_dict_tag=aruco_dict_tag,
        )

        self = cls(**init_kwargs)
        self.init_kwargs = init_kwargs
        self.calibration_img = cv2.cvtColor(
            self.board.draw(hw[::-1]), cv2.COLOR_GRAY2RGB
        )
        self.calibration_img_info = dict(hw=hw, ppi=ppi, **init_kwargs)
        self.calibration_img_name = (
            f"charuco_square_x{square_xy[0]}y{square_xy[1]}_size{size_mm}mm.png"
        )
        return self

    def __getstate__(self):
        self.aruco_dictionary = None
        return

    def __setstate__(self, state=None):
        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(self.aruco_dict_tag)


if __name__ == "__main__":
    from boxx import *

    feature_lib = CharucoFeatureLib.build_with_calibration_img(
        aruco_dict_tag="DICT_4X4_250_start50"
    )
    calibration_img = img = feature_lib.calibration_img
    # clip to 100 for printer to use less black ink
    boxx.imsave(f"/tmp/{feature_lib.calibration_img_name}", calibration_img.clip(100,))
    img = np.pad(calibration_img, ((50, 50), (50, 50), (0, 0)), constant_values=255)
    d = dict(img=img)
    # feature_lib = CharucoFeatureLib(aruco_dict_tag = cv2.aruco.DICT_APRILTAG_25H9)
    feature_lib.find_image_points(d)
    vis = feature_lib.vis(d)

    boxx.tree(d, deep=1)
    boxx.show(vis)
