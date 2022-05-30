# `calibrating`: A Python Library for Camera Calibration
Calibrate the internal and external parameters of cameras, rectify stereo cameras by OpenCV python.

[![stereo-checkboard](https://user-images.githubusercontent.com/10448025/131808105-a325961e-5fbb-4475-adcd-ba0e2c53e268.png)](https://yl-data.github.io/2108.calibrating-vis/stereo/index.html)
<!-- ![stereo](https://user-images.githubusercontent.com/10448025/131805868-e73cd022-d79b-400c-b057-c26915f92c7c.jpg) -->

## ▮ Features
- [High-level API](example/checkboard_example.py) that simplifies calibration steps
- Rich visualization to verify the calibration effect. e.g. [stereo-rectify-vis](https://yl-data.github.io/2108.calibrating-vis/stereo/index.html), [reproject-depth-vis](https://yl-data.github.io/2108.calibrating-vis/project-depth/index.html)
- Very easy to install and run example
- Mature [stereo module](calibrating/stereo.py) for correctly calculate the depth
- Decoupling the feature extraction and calibration process, support both checkboard and markers(`cv2.aruco`)
- Provide camera internal and external parameters standard, which can be exported as `.yaml`
- Support for [occluded marker](example/test_occlude_marker.py)
- Automatically ignore non-compliant images
## ▮ Install
```bash
pip3 install calibrating
```
## ▮ Run Example
Example images are captured by paired_stereo_and_depth_cams:   
[![paired_stereo_and_depth_cams_1](https://user-images.githubusercontent.com/10448025/131831496-7a38c677-a578-4a4e-a01e-aa102dad9dbc.jpg)](https://github.com/yl-data/calibrating_example_data/raw/master/paired_stereo_and_depth_cams.jpg?raw=true)

```bash
pip3 install calibrating
# Prepare example data(100MB): checkboard images of paired stereo and depth cameras
git clone https://github.com/yl-data/calibrating_example_data

# Prepare example code
git clone https://github.com/DIYer22/calibrating

# Run checkboard example 
python3 calibrating/example/checkboard_example.py
```
Finally, your browser will open [stereo-rectify-vis](https://yl-data.github.io/2108.calibrating-vis/stereo/index.html), [reproject-depth-vis](https://yl-data.github.io/2108.calibrating-vis/project-depth/index.html)


Detailed example code with comments: [example/checkboard_example.py](example/checkboard_example.py)   
Or Chinese Version: [example/checkboard_example_cn.py (中文注释)](example/checkboard_example_cn.py)

## ▮ Stereo

Run [stereo example](calibrating/stereo.py):
```bash
python3 calibrating/calibrating/stereo.py
```
After a while, your browser will open:
- [stereo-rectify-vis](https://yl-data.github.io/2108.calibrating-vis/stereo/index.html)
- [StereoSGBM-depth-vis](https://yl-data.github.io/2108.calibrating-vis/stereo_sgbm_vis/): Which shows `undistort_img1`, `unrectify_depth`. The example disparity is computed by `cv2.StereoSGBM`.

**[Mermaid](https://mermaid.live/) flowchart of `calbrating.Stereo.get_depth(img1, img2)`**
```mermaid
flowchart 
    subgraph "Stereo.get_depth(img1, img2)"
        input(input:　\nimg1\nimg2)--> undistort
        undistort-->rectify
        undistort --> stereo_re
        subgraph StereoMatching
        end
        rectify --> StereoMatching
        StereoMatching --> disparity_to_depth
        disparity_to_depth --> unrectify
        unrectify --> stereo_re("output:　　　　\nrectify_img1\nrectify_depth")
    end
```


