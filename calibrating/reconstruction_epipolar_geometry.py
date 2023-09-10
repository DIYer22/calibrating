import boxx
import numpy as np
from numpy.linalg import inv

with boxx.inpkg():
    from .flow_utils import flow_normal_to_abs
    from .epipolar_geometry import matching_uvs_in_one_img, EssentialMatrixStereo


"""      
算法思路:       
需求:
1. 任何视角至少和一个其它视角有 common view, 且视角间构成的 common view 邻接矩阵全联通, 只有一个子图
2. depth 尺度能被传导, 且 depth 尺度能传导到所有视角. depth 尺度传导的条件, 对于 v1 v2 v3:
    - matching_uvs_in_one_img(v2_of_12, v2_of_23) 有匹配, 这样 v1 v2 v3 尺度就能一致. 因此满足条件 2 则必然满足条件 1
进阶需求:
- [ ] 后端联合优化

步骤:
- 对每个 view, 获得和其它所有 view 的 匹配 uvs(把两张 flow 转换为 uvs)
- 对于每个 set(v 三元组) from C_n_3, 选择 min(另外两个 uvs 点)最大的作为 matching_uvs_in_one_img 主场, 求 uvs 匹配点
- 选取出匹配点最多的作为种子, 且以其 matching_uvs_in_one_img 主场 view 作为世界坐标系
- 对每个没聚合的 v三元组, 从 set 差一的三元组中 选 matching_uvs_in_one_img 最大的加入图
- 同时, 每个加入三元组给新加入的那个 view 计算 T_world
- 在 set 差一的元素中 删除包含刚加入 view 的三元组(全都加入了图)
- 重复直到没有set 差一的三元组
- 若还存在三元组, 则报错多个子图或者重新找种子
"""


class ReconstructionExtrinsics:
    def __init__(self, viewds, set2ds=None, flowds=None, cfg=None):
        """
        根据多视角的匹配好的关键点(或光流)求解多个视角的相对相机外参
        Input 参数:
            viewds 是 dict, 表示每个视角(view)各自包含的信息
            其 key 是视角编号, value 是 dict, 0 号视角的 value 结构如下:
            >>> tree(viewds[0])
                └── /: dict
                    ├── K: (3, 3)float32          # 必选, 该视角的相机内参
                    ├── mask: (1024, 1024)bool    # 可选, instance mask 用于 debug
                    └── T: (4, 4)float64          # 可选, 该视角的 GT 相机外参

            set2ds 是 dict, 可选, 表示两个视角组合下包含的信息, 比如两个视角中匹配成功的关键点
            其 key 是两个视角的 frozenset, 由于 a=>b 和 b=> a 是等价的, 所以用 frozenset 来作为 key
            其 value 是 dict, 比如 [0, 1] 两个视角的 value 结构如下:
            >>> tree(set2ds[frozenset([0, 1])])
                └── /: dict
                    ├── uvs_i: (59601, 2)float64  # 被匹配上的点在 view_i 中的 uv 坐标, i 表示 list(key)[0] 的 view
                    └── uvs_j: (59601, 2)float64  # 被匹配上的点在 view_j 中的 uv 坐标, j 表示 list(key)[1] 的 view

            flowds 是 dict, 可选, 表示 view_a 到 view_b 的光流
            其 key 是 tuple(a, b), key 可同时存在 (a, b) 和 (b, a)
            其 value 是 dict, 比如 view0 到 view1 的 value 结构如下:
            >>> tree(flowds[(0, 1)])
                └── /: dict
                    ├── flow_abs: (1024, 1024, 2)float32        # 可选, 和 flow_normal 存在一个即可
                    ├── flow_normal: (2, 1024, 1024)float32     # 可选, 和 flow_abs 存在一个即可
                    └── common_fov_mask: (1024, 1024)bool       # 必选, view0, view1 共同视野的 mask, 即 view0 中能被 view1 也看见的像素

            重建只需要 viewds 和 set2ds
            但当输入 flowds 时, 会自动生成 set2ds

        Output:
            会把每个 view 的计算结果存放回 viewds, 会新增 "T_re" 和 "uvzis", 如下:
            >>> tree(viewds[0])
                └── /: dict
                    ├── K: (3, 3)float32                # 输入的内参
                    ├── T_re: (4, 4)float64             # 计算出每个 view 相机在世界坐标系的外参
                    └── uvzis: (60888, 4)float64        # 从匹配点中计算出的深度, [u:像素横坐标, v:像素纵坐标, z:深度, i:被匹配的 view 编号]
        """
        self.cfg = cfg or {}
        self.viewds = viewds
        if not set2ds and flowds:
            set2ds = self.build_set2ds_by_flowds(viewds, flowds)
        self.set2ds = set2ds
        self.flowds = flowds

        # build_set3ds
        from itertools import combinations

        def gen_set3_dic(idx_sorted, set3d=None):
            set3d = set3d or {}
            set3d["idx_sorted"] = idx_sorted
            ii, jj, kk = idx_sorted

            set3d["uvsd"] = uvsd = {}
            for idx_other in (jj, kk):
                set2 = frozenset([idx_other, ii])
                ij = tuple(sorted(set2))
                suffix = "ij"[ij.index(ii)]
                uvs_main = set2ds[set2][f"uvs_{suffix}"]

                uvs_other = set2ds[set2][f"uvs_{'ji'[ij.index(ii)]}"]
                uvsd[idx_other] = dict(uvs_main=uvs_main, uvs_other=uvs_other)

            matched = matching_uvs_in_one_img(
                uvsd[jj]["uvs_main"], uvsd[kk]["uvs_main"]
            )
            # jj is uv_match_idx1, kk is uv_match_idx2
            if not matched:
                return
            set3d.update(matched)
            if flowds and boxx.mg():
                mask_ik = flowds[(ii, kk)]["common_fov_mask"]
                mask_inst = (
                    viewds[ii]["mask"]
                    if "mask" in viewds[ii]
                    else np.ones_like(mask_ik)
                )
                trio_view_mask = (
                    mask_inst + mask_ik * 2 + flowds[(ii, jj)]["common_fov_mask"]
                )
                trio_view = boxx.mapping_array(
                    trio_view_mask,
                    [
                        (0, 0, 0),
                        (128, 128, 128),
                        (255, 0, 0),
                        (0, 255, 255),
                        (255, 255, 255),
                    ],
                )
                # print(Counter(trio_view_mask.flatten()));tree(matched)
                set3d["trio_view"] = np.uint8(trio_view)
                set3d["trio_view_mask"] = np.uint8(trio_view_mask)
            return set3d

        set3ds = {}
        for set3 in list(combinations(viewds, 3)):
            set3 = frozenset(set3)
            ijk = tuple(sorted(set3))
            not_include_uvsn = {
                idx: len(set2ds.get(set3.difference({idx}), {"uvs_i": ""})["uvs_i"])
                for idx in ijk
            }
            if list(not_include_uvsn.values()).count(0) >= 2:
                break

            idx_sorted = sorted(ijk, key=lambda x: not_include_uvsn[x])
            set3d = gen_set3_dic(idx_sorted)
            if set3d:
                set3d["not_include_uvsn"] = not_include_uvsn
                set3ds[set3] = set3d

        # generate_propagate_path
        set3_sort_matched = sorted(
            set3ds, key=lambda x: len(set3ds[x]["uv_match_idx1"])
        )[::-1]
        seed = set3_sort_matched.pop(0)
        propagated = set(seed)
        propagate_path = [
            (set3ds[seed]["idx_sorted"][2], seed),
        ]  # [(idx_new, set3),]
        while len(set3_sort_matched):
            for set3 in set3_sort_matched[:]:
                diff = set3.difference(propagated)
                if len(diff) == 1:
                    idx_new = list(diff)[0]
                    set3d = set3ds[set3]
                    set3_sort_matched.remove(set3)
                    if idx_new == set3d["idx_sorted"][0]:
                        # 如果新加入的 view 是主相机, 则无法传播
                        if set3d["not_include_uvsn"][idx_new] == 0:
                            break  # 如果另外两个相机间没有光流匹配, 则set3作废
                        # 需要换已加入的相机作为主相机才能在两个 stereo 中传播
                        set3ds[set3] = gen_set3_dic(
                            set3d["idx_sorted"][1:] + set3d["idx_sorted"][:1], set3d
                        )

                    propagated = propagated.union(set3)
                    propagate_path.append((idx_new, set3))
                    break
                if len(diff) == 0:
                    set3_sort_matched.remove(set3)
                    break

            if len(diff) in [2, 3]:
                raise Exception("Has two subgraph")

        # propagate_scale
        stereods = {}
        idx_seed = set3ds[seed]["idx_sorted"]
        idx_seed_main = idx_seed[0]
        idx_seed_2th = idx_seed[1]
        viewds[idx_seed_main]["T_re"] = np.eye(4)
        propagate_baseline = {}
        # viewds[idx_seed_main]["T_re"] = viewds[idx_seed_main]["T"]
        # t_seed2 = T_to_deg_distance(viewds[idx_seed_main]["T"], viewds[idx_seed_2th]["T"])["distance"]
        # propagate_baseline = {frozenset([idx_seed_main, idx_seed_2th]):t_seed2}
        for idx_new, set3 in propagate_path + [(idx_seed_2th, seed)]:
            set3d = set3ds[set3]
            assert idx_new != set3d["idx_sorted"][0]
            idx_main, jj, kk = set3d["idx_sorted"]
            idx_propagated = kk if idx_new == jj else jj
            uvsd = set3d["uvsd"]

            def get_stereo(ii, jj):
                if (ii, jj) not in stereods:
                    stereods[(ii, jj)] = EssentialMatrixStereo(
                        uvsd[jj]["uvs_main"],
                        uvsd[jj]["uvs_other"],
                        K1=viewds[ii]["K"],
                        K2=viewds[jj]["K"],
                        xy1=viewds[ii]["img"].shape[:2][::-1],
                        xy2=viewds[jj]["img"].shape[:2][::-1],
                        name1=ii,
                        name2=jj,
                        baseline=propagate_baseline.get(frozenset([ii, jj]), 1),
                    )
                return stereods[(ii, jj)]

            stereo_new = get_stereo(idx_main, idx_new)
            stereo_propagated = get_stereo(idx_main, idx_propagated)
            suffix_new, suffix_propagated = "12" if idx_new == jj else "21"
            matched = dict(
                uv_match_idx1=set3d["uv_match_idx" + suffix_new],
                uv_match_idx2=set3d["uv_match_idx" + suffix_propagated],
            )
            stereo_new.align_scale_with(stereo_propagated, matched)
            propagate_baseline[frozenset([idx_main, idx_new])] = stereo_new.baseline
            viewds[idx_new]["T_re"] = viewds[idx_main]["T_re"] @ inv(stereo_new.T)

        # get depth and set scale to depth near 1 and view[0] as R
        for (k1, k2), stereo in stereods.items():
            d = stereo.epipolar
            d1 = viewds[k1]
            uvzis = np.concatenate(
                (d["uvs1"], d["zs1"][:, None], [[k2]] * d["zs1"].size), -1
            )
            d1["uvzis"] = (
                np.concatenate((uvzis, d1["uvzis"])) if "uvzis" in d1 else uvzis
            )

            d2 = viewds[k2]
            uvzis = np.concatenate(
                (d["uvs2"], d["zs2"][:, None], [[k1]] * d["zs2"].size), -1
            )
            d2["uvzis"] = (
                np.concatenate((uvzis, d2["uvzis"])) if "uvzis" in d2 else uvzis
            )

        z_mean = np.concatenate([d["uvzis"][:, 2] for d in viewds.values()]).mean()
        self.change_scale(1 / z_mean)
        T0_target = np.eye(4)
        T0_target[2, 3] = -viewds[0]["uvzis"][:, 2].mean()
        self.apply_T(T=T0_target @ inv(viewds[0]["T_re"]))

        self.set3ds = set3ds
        self.seed = seed
        self.propagate_path = propagate_path
        self.stereods = stereods
        boxx.mg()

    @staticmethod
    def build_set2ds_by_flowds(viewds, flowds):

        set2ds = {}
        for set2 in set(map(frozenset, flowds)):
            set2d = {}
            ij = tuple(sorted(set2))

            def process_flow_to_uvs(xx, name="ij"):
                if xx in flowds and flowds[xx]["common_fov_mask"].sum() > 10:
                    flow_abs = flowds[xx].get("flow_abs")
                    mask = flowds[xx]["common_fov_mask"]
                    if flow_abs is None:
                        flow_normal = flowds[xx]["flow_normal"]
                        target_shape = (
                            viewds[xx[-1]]["mask"].shape
                            if ("mask" in viewds[xx[-1]])
                            else flow_normal.shape
                        )
                        flow_abs = flow_normal_to_abs(flow_normal, target_shape)
                    # TODO 思考要不要加 0.5?  np.round(np.arange(100)+0.5) is [0, 2, 2, 4, 4, 6, 6, 8, 8...
                    h, w = flow_abs.shape[:2]
                    xys_abs = (np.mgrid[:h, :w] + 0.5 - 1e-8)[::-1].transpose(1, 2, 0)
                    set2d.update(
                        {
                            f"uvs_{name}_{name[0]}": xys_abs[mask],
                            f"uvs_{name}_{name[1]}": (flow_abs + xys_abs)[mask],
                        }
                    )

            process_flow_to_uvs(ij, "ij")
            process_flow_to_uvs(ij[::-1], "ji")
            if len(set2d):
                set2d["uvs_i"] = np.concatenate(
                    [set2d[k] for k in set2d if k in ["uvs_ij_i", "uvs_ji_i"]]
                )
                set2d["uvs_j"] = np.concatenate(
                    [set2d[k] for k in set2d if k in ["uvs_ij_j", "uvs_ji_j"]]
                )
                set2ds[set2] = set2d
        return set2ds

    def change_scale(self, rate=1, T=None):
        """
        必须先做到尺度一致后 T_gt@inv(T_re) 才能有效, 因为 T 只做线性变换
        -所以得分两步做:
        1. 尺度一致 2. 根据尺度一致的 T_re 和 T_gt 计算 T
        """
        for viewd in self.viewds.values():
            if T is None:
                viewd["uvzis"][:, 2] *= rate
                viewd["T_re"][:3, 3] *= rate
            else:
                assert rate == 1
                viewd["T_re"] = T @ viewd["T_re"]

        return self.viewds

    def apply_T(self, T):
        return self.change_scale(T=T)
