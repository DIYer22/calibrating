#!/usr/bin/env python3
import cv2
import boxx
from boxx import np

with boxx.inpkg():
    from . import interpolate_uvzs, interpolate_sparse2d, uvzs_to_arr2d


class MetaStereoMatching:
    def __init__(self, cfg=None):
        self.cfg = cfg

    def __call__(self, img1, img2):
        # input: RGB uint8 (h, w, 3)uint8
        raise NotImplementedError()
        # output: float disparity (h, w)float64, unit is m
        # return disparity or dict(disparity=disparity)


# A Example of StereoMatching class
class SemiGlobalBlockMatching(MetaStereoMatching):
    def __init__(self, cfg=None):
        if cfg is None:
            cfg = {}
        self.cfg = cfg
        self.max_size = self.cfg.get("max_size", 1000)

        # StereoSGBM_create from https://gist.github.com/andijakl/ffe6e5e16742455291ef2a4edbe63cb7
        block_size = 11
        min_disp = 2
        max_disp = 220
        # Maximum disparity minus minimum disparity. The value is always greater than zero.
        # In the current implementation, this parameter must be divisible by 16.
        num_disp = max_disp - min_disp
        # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
        # Normally, a value within the 5-15 range is good enough
        uniquenessRatio = 5
        # Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
        # Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
        speckleWindowSize = 200
        # Maximum disparity variation within each connected component.
        # If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
        # Normally, 1 or 2 is good enough.
        speckleRange = 2
        disp12MaxDiff = 0

        self.stereo_sgbm = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            uniquenessRatio=uniquenessRatio,
            speckleWindowSize=speckleWindowSize,
            speckleRange=speckleRange,
            disp12MaxDiff=disp12MaxDiff,
            P1=8 * 1 * block_size * block_size,
            P2=32 * 1 * block_size * block_size,
        )

    def __call__(self, img1, img2):
        resize_ratio = min(self.max_size / max(img1.shape[:2]), 1)
        simg1, simg2 = boxx.resize(img1, resize_ratio), boxx.resize(img2, resize_ratio)
        sdisparity = (self.stereo_sgbm.compute(simg1, simg2).astype(np.float32)).clip(0)
        sdisparity[sdisparity < self.stereo_sgbm.getMinDisparity() * 16] = 0
        disparity = (
            boxx.resize(sdisparity / 16.0, img1.shape[:2])
            * img1.shape[1]
            / simg1.shape[1]
        )
        return disparity


class MatchingByBoard(MetaStereoMatching):
    """
    Stereo mathch disp by board's image_points
    return a dense disp on calibration board
    """

    def __init__(self, board, dense_predict=True):
        self.board = board
        self.dense_predict = dense_predict

    def __call__(self, img1, img2):
        self.board
        d1 = dict(img=img1)
        self.board.find_image_points(d1)
        image_points1 = d1["image_points"]
        d2 = dict(img=img2)
        self.board.find_image_points(d2)
        image_points2 = d2["image_points"]

        if isinstance(image_points1, dict):
            image_points1 = []
            image_points2 = []
            for key in sorted(set(d1["image_points"]).intersection(d2["image_points"])):
                image_points1.append(d1["image_points"][key])
                image_points2.append(d2["image_points"][key])
            image_points1 = np.concatenate(image_points1, 0)
            image_points2 = np.concatenate(image_points2, 0)

        rectify_std = np.std((image_points2 - image_points1)[:, 1])
        point_disps = (image_points1 - image_points2)[:, 0]
        xyds = np.append(image_points1, point_disps[:, None], axis=-1)
        sparse_disp = uvzs_to_arr2d(xyds, img1.shape[:2])
        if self.dense_predict:
            dense_disp = 1 / interpolate_sparse2d(1 / sparse_disp, "convex_hull")
            disparity = dense_disp
        else:
            disparity = sparse_disp
        return dict(disparity=disparity, rectify_std=rectify_std)


class FeatureMatchingAsStereoMatching(MetaStereoMatching):
    def __init__(self, feature_matching):
        self.feature_matching = feature_matching

    def __call__(self, img1, img2):
        matched = self.feature_matching(img1, img2)
        hw = img1.shape[:2]
        resize_shape = self.feature_matching.cfg.get("shape", hw)
        resize_shape = (
            resize_shape[0] // 8,
            resize_shape[1] // 8,
        )
        uvs1, uvs2 = (
            matched["uvs1"] * resize_shape[::-1],
            matched["uvs2"] * resize_shape[::-1],
        )

        uvds = np.concatenate((uvs1, (uvs1 - uvs2)[:, :1]), 1)
        disparity = interpolate_uvzs(
            uvds, resize_shape, constrained_type=None, inter_type="nearest"
        )
        if resize_shape != hw:
            # resize time 767s => 44s
            # p=1 => 42s
            # //8 => 0.530689
            # //4 => 2.5
            disparity = disparity * hw[1] / resize_shape[1]
            disparity = boxx.resize(disparity, hw, cv2.INTER_NEAREST)
        boxx.mg()
        return dict(disparity=disparity, matched=matched)


if __name__ == "__main__":
    from boxx import *
    import calibrating
    from calibrating import vis_depth

    example_type = "checkboard"
    example_type = "aruco"
    caml, camr, camd = calibrating.get_test_cams(example_type).values()

    key = list(caml)[0]
    img1 = boxx.imread(caml[key]["path"])
    img2 = boxx.imread(camr[key]["path"])

    stereo = calibrating.Stereo(caml, camr)

    stereo_matching_sgbm = SemiGlobalBlockMatching()

    import calibdiff  # need pip install calibdiff

    feature_matching = calibdiff.LoftrFeatureMatching(dict(shape=(480, 640)))
    stereo_matching_lofter = FeatureMatchingAsStereoMatching(feature_matching)

    stereo_matching_board = MatchingByBoard(caml.board)

    stereo.set_stereo_matching(stereo_matching_sgbm)
    depth_re = stereo.get_depth(img1, img2)
    depth = depth_re["unrectify_depth"]

    calibrating.vis_align(depth_re["undistort_img1"], depth_re["unrectify_depth"])

    boxx.shows(
        depth_re["undistort_img1"],
        [
            vis_depth(
                stereo.set_stereo_matching(sm).get_depth(img1, img2)["unrectify_depth"],
                fix_range=(1, 3.5),
            )
            for sm in [
                stereo_matching_sgbm,
                stereo_matching_lofter,
                stereo_matching_board,
            ]
        ],
    )

    # showb-feature_matching.vis(matched)
    # showb-vis_depth(depth, fix_range=(1, 3.5))
