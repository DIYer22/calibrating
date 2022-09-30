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
- [x] stereo 包含两个 cam
- [x] ~~stereo 应该继承自 Cams?~~
- [x] stereo dump yaml
- [x] ~~stereo 接口和 nn rectify, nn undisort 兼容~~
    - 暂时不考虑
- [x] 能处理好畸变和 rectify 
- [x] 我求的外参和 stereoCalibrate 求的外参有毫米级别的不一致
    - 以 stereoCalibrate 为准
- [x] 很难兼容原来的接口了?
    - 以前用途不广, 能接受破坏性升级
- [x] 有两组内外参该怎么办?
    - 以 stereoCalibrate 为准

## TODO list
- [ ] 新的科学的测试数据
    - 拍平文件夹, 方便查看
    - 有 multi board example, for reconstruction
    - 讲清楚棋盘格的缺点, 旋转对称, 遮挡, 求 feature 慢
- [ ] 智能提示:
    - 根据标定板 Ts 的丰富程度(Ts 和 meanT 的 R 分布大)和 image point 覆盖程度智能提示
    - vis_image_points_cover 添加 assert 和 warnning, 及开关
- [ ] rename feature_lib => board
- [ ] cam.rotate 考虑畸变
- [ ] 更完善的代码文档, 尤其是 d 和 MetaFeatureLib
- [ ] 自动测试用例
## Done
- 和建冉沟通 calibrating 后
    - 目的: 抽象出来并共用 Cam 模型
    - [ ] ~~去掉 boxx 强依赖~~
        - 大量可视化, 去不掉, 只能考虑未来 mxcv 出来
    - [x] 新增feature: chekboard + marker
    - [x] 兼容 py3.6
    - [x] cam 支持预定义内参 K, D
    - [x] dumap and load yaml of KD
- [x] support Occlusion for external parameters(支持被遮挡的 marker)
- [x] `aruco` 的 `caml.vis_depth_alignment(imgl, depthl)` 存在轻微没对齐
    - 结论: 本来 depthd 在标定板边缘就有深度缺失
    - 结论2: 用梅卡的内参反而不准了
- [x] Stereo different intrinsic cams
- [x] depth_to_disparity, disparity_to_depth
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

