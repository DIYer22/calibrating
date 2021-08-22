#!/usr/bin/env python3
import cv2
import boxx
import pickle
import numpy as np


class Stereo:
    @classmethod
    def init_by_calibrate(cls, K1, D1, K2, D2, xy, object_points, l1, l2):
        K2, D2 = K1, D1
        # Compute Extrinsics Matrix
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC

        stereocalib_criteria = (
            cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
            100,
            1e-5,
        )
        ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            object_points,
            l1,
            l2,
            K1,
            D1,
            K2,
            D2,
            xy,
            criteria=stereocalib_criteria,
            flags=flags,
        )
        return cls().init_by_K_Rt(K1, D1, K2, D2, xy, R, T)

    def init_by_K_Rt(self, K1, D1, K2, D2, xy, R, T):
        R1, R2, P1, P2, Q, roimg1, roimg2 = cv2.stereoRectify(K1, D1, K2, D2, xy, R, T)
        self.data = dict(K1=K1, D1=D1, R1=R1, K2=K2, D2=D2, R2=R2, xy=xy, R=R, T=T)
        self._set_map()
        return self

    def _set_map(self):
        data = self.data
        xy = data["xy"]
        K1 = data["K1"]
        K2 = data["K2"]
        R1 = data["R1"]
        R2 = data["R2"]
        D1 = data["D1"]
        D2 = data["D2"]

        self.l_maps = cv2.initUndistortRectifyMap(K1, D1, R1, K1, xy, cv2.CV_32FC1)
        self.r_maps = cv2.initUndistortRectifyMap(K2, D2, R2, K2, xy, cv2.CV_32FC1)

    @staticmethod
    def _get_img(path_or_np):
        if isinstance(path_or_np, str):
            return cv2.imread(path_or_np)[..., ::-1]
        return path_or_np

    def align(self, pair):
        ir1 = cv2.remap(
            self._get_img(pair[0]), self.l_maps[0], self.l_maps[1], cv2.INTER_LANCZOS4
        )
        ir2 = cv2.remap(
            self._get_img(pair[1]), self.r_maps[0], self.r_maps[1], cv2.INTER_LANCZOS4
        )
        return [ir1, ir2]

    @staticmethod
    def vis(pair, n_line=21, thickness=0.03):
        """
        Draw lines on stereo image pairs, two cases of horizontal align and vertical align.
        Input: image numpy pairs.
        """
        colors = [
            [255, 0, 0],
            [0, 255, 255],
            [0, 255, 0],
            [255, 0, 255],
            [0, 0, 255],
            [255, 255, 0],
        ]
        vis = np.concatenate(pair, 1)
        img_size = pair[0].shape[0]
        _thickness = max(1, int(round(thickness * img_size / (n_line + 1))))
        gap = img_size / (n_line + 1)
        for i in range(n_line):
            b = int((i + 1) * gap - _thickness / 2)
            vis[b : b + _thickness] = colors[i % len(colors)]
        return vis

    @classmethod
    def shows(cls, *l, **kv):
        vis = cls.vis(*l, **kv)

        idx = vis.shape[1] // 2
        viss = [vis[:, :idx], vis[:, idx:]]
        boxx.shows(viss)
        return viss

    def save(self, pkl_path="stereo_calibration.pkl"):
        """
        Save calibration object as pickle file.
        """
        pickle.dump(self.data, open(pkl_path, "wb"))
        return pkl_path

    def load(self, pkl_path="stereo_calibration.pkl"):
        """
        Load calibration object from pickle file.
        """
        if isinstance(self, str):
            pkl_path = self
            self = Stereo()
        self.data = pickle.load(open(pkl_path, "rb"))
        self._set_map()
        return self


if __name__ == "__main__":
    pass
