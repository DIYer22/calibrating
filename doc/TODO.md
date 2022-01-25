# TODO

## 一些想法
- Deal defect depth2
    - 主要是旁边的 depth 通过深度缺失口映射为了新的错误 depthl
    - 方案1: 在 depthl 上, 射线上的点若远于对应的 depth, 也 mask 掉
    - 方案2: 深度补全然后
    - 方案3: 预测的地方在 camd 坐标下是深度缺失, 则抛弃
- 有点后悔了以标定图片为中心了, 应该以相机模型为中心
    - dict 放的不是标定图片
    - 希望 init 改掉 标定图像为中心
    - 先就这样吧, 反正都已经支持了 cam.load
    - vis 需要输入原图

## tmp
期望
- stereo 包含两个 cam
- stereo 应该继承自 Cams?
- stereo dump yaml
- stereo 接口和 nn rectify, nn undisort 兼容
- 能处理好畸变和 rectify 
- 我求的外参和 stereoCalibrate 求的外参有毫米级别的不一致
- 很难兼容原来的接口了?
- 有两组内外参该怎么办?
- BUG
    - [ ] 无法直接跑通
    - [ ] dump load 挂了

## TODO list
- [ ] depth_to_disparity, disparity_to_depth
- [ ] Stereo different intrinsic cams
- [ ] support Occlusion for external parameters
- [ ] switch between depth and disparity
- 和建冉沟通 calibrating 后
    - [ ] 目的: 抽象出来并共用 Cam 模型
    - [ ] 支持被遮挡的 marker
    - [ ] 新增feature: chekboard + marker
    - [ ] 支持获得单板外参 Cam.get_calibration_board_T
    - [ ] 期望支持单个 Marker 外参
    - [ ] 兼容 py3.6
    - [ ] 去掉 boxx 强依赖
    - [x] cam 支持预定义内参 K, D
    - [x] dumap and load yaml of KD
## Done
- [x] init from known K, D
- [x] English doc
- [x] Examples and better example 
- 全部以 m 为单位
- [x] 以 key: imgp or imgps 为项
- [x] 支持棋盘格, aruco
- [x] 自动剔除不达标的项
- [x] 自动在 '/tmp' 下保存可视化, 并方便查看
- [x] 支持缓存
- [x] 丰富的可视化
- [x] depth point cloud interpolation

