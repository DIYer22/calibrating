# TODO list
- [ ] depth_to_disparity, disparity_to_depth
- [ ] Stereo different intrinsic cams
- [ ] support Occlusion for external parameters
- [ ] Deal defect depth2
    - 主要是旁边的 depth 通过深度缺失口映射为了新的错误 depthl
    - 方案1: 在 depthl 上, 射线上的点若远于对应的 depth, 也 mask 掉
    - 方案2: 深度补全然后
    - 方案3: 预测的地方在 camd 坐标下是深度缺失, 则抛弃

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

