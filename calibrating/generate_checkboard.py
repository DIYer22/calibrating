#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import boxx
import numpy as np

if __name__ == "__main__":
    # TODO better cli
    # A3
    width = 3508
    height = 4961
    n = 15
    ppi = 300  # pixels per inch (ppi)

    size = height // (n + 1)
    size_mm = round(size / ppi * 25.4, 2)

    hn = int((height - size) / size)
    wn = int((width - size) / size)

    # 长宽需要互为奇偶, 防止旋转对称导致的多个合法外参
    wn -= 1 - (hn % 2)

    img = np.ones((height, width), dtype=np.uint8) * 255
    print(img.shape[0], img.shape[1])

    for hidx in range(hn):
        for widx in range(wn):
            color = [0, 255][(hidx + widx) % 2]
            p = [hidx * size + size // 2, widx * size + size // 2]
            img[p[0] : p[0] + size, p[1] : p[1] + size] = color

    fname = f"checkboard_hw{hn-1}x{wn-1}_size{size_mm}mm.png"
    print("Save to:", fname)
    boxx.imsave(fname, img)
    boxx.showb(fname)
