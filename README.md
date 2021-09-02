# `calibrating`: A camera calibration library
Calibrate the internal and external parameters of cameras, rectify stereo cameras by OpenCV python.

![stereo-checkboard](https://user-images.githubusercontent.com/10448025/131808105-a325961e-5fbb-4475-adcd-ba0e2c53e268.png)
<!-- ![stereo](https://user-images.githubusercontent.com/10448025/131805868-e73cd022-d79b-400c-b057-c26915f92c7c.jpg) -->

## ▮ Features
- High-level API that simplifies calibration steps
- Decoupling the feature extraction and calibration process, support both checkboard and markers(`cv2.aruco`)
- Automatically ignore non-compliant images
- Rich visualization. e.g. [stereo-recitfy-vis](https://yl-data.github.io/2108.calibrating-vis/stereo/index.html), [reproject-depth-vis](https://yl-data.github.io/2108.calibrating-vis/project-depth/index.html)

## ▮ Install
```bash
pip install calibrating
```
## ▮ Run Example
Example images are captured by paired_stereo_and_depth_cams:   
 ![](https://github.com/yl-data/calibrating_example_data/blob/master/paired_stereo_and_depth_cams.jpg?raw=true)

```bash
# Prepare example data: checkboard images of paired stereo and depth cameras
git clone https://github.com/yl-data/calibrating_example_data

# Prepare example code
git clone https://github.com/DIYer22/calibrating

# Run checkboard example 
python calibrating/example/checkboard_example.py
```
Finally, your browser will open [stereo-recitfy-vis](https://yl-data.github.io/2108.calibrating-vis/stereo/index.html), [reproject-depth-vis](https://yl-data.github.io/2108.calibrating-vis/project-depth/index.html)


Detailed example code with comments: [example/checkboard_example.py](example/checkboard_example.py)   
Or Chinese Version: [example/checkboard_example_cn.py (中文注释)](example/checkboard_example_cn.py)



<style>

img{
    max-height:350px;
}
</style>
