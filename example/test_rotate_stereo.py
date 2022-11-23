#!/usr/bin/env python3
import os
import cv2
import boxx


def rotate_right_img_imread(fname, *l, **kv):
    import skimage.io

    img = skimage.io.imread(fname, *l, **kv)
    if "_r.jpg" in fname:
        img = np.rot90(img, 2)
    return img


boxx.imread = rotate_right_img_imread
imread = rotate_right_img_imread

if __name__ == "__main__":
    """
    Verify that right eye rotation 180° causes problems when using cv2.stereoRectify()
    """
    from boxx import *

    os.system("rm /tmp/calibrating-cache-camr.pkl")
    from test_different_stereo import *

    os.system("rm /tmp/calibrating-cache-camr.pkl")

    class StereoWithCv2Rectify(Stereo):
        """When the right eye camera is rotated 180° around the z axis, a BUG will appear"""

        stereo_recitfy = Stereo.stereo_recitfy_by_cv2

    stereo = Stereo(caml, camr)
    stereo_cv2 = StereoWithCv2Rectify(caml, camr)

    visn = 1
    Cam.vis_stereo(caml, camr, stereo, visn)
    Cam.vis_stereo(caml, camr, stereo_cv2, visn)

    stereo_matching = SemiGlobalBlockMatching({})
    [
        s.set_stereo_matching(
            stereo_matching, max_depth=3, translation_rectify_img=True
        )
        for s in [
            stereo,
            stereo_cv2,
        ]
    ]

    depths = [
        s.get_depth(imgl, imgr)["unrectify_depth"]
        for s in [
            stereo,
            stereo_cv2,
        ]
    ]

    boxx.shows([undistort_imgl, list(map(vis_depth, depths))])
