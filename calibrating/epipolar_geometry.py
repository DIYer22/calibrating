#!/usr/bin/env python3

import boxx
import numpy as np

with boxx.inpkg():
    from .camera import Cam
    from .stereo_camera import Stereo
    from .utils import inv, R_t_to_T, apply_T_to_point_cloud


def compute_essential_matrix(xyzs1, xyzs2):
    assert xyzs1.shape == xyzs2.shape, "The shapes of input points must be the same."
    n = xyzs1.shape[0]
    assert n >= 8, "At least 8 point pairs are required."
    if n > 100:
        xyzs1 = xyzs1[:: n // 100]
        xyzs2 = xyzs2[:: n // 100]
        n = xyzs1.shape[0]
    # 构建矩阵A
    A = np.zeros((n, 9))
    for i in range(n):
        x1, y1, z1 = xyzs1[i]
        x2, y2, z2 = xyzs2[i]
        A[i] = [
            x1 * x2,
            x2 * y1,
            z1 * x2,
            x1 * y2,
            y1 * y2,
            z1 * y2,
            x1 * z2,
            y1 * z2,
            z1 * z2,
        ]
    # 使用SVD分解求解线性方程组Ax = 0
    _, _, Vt = np.linalg.svd(A)
    E = Vt[-1].reshape(3, 3)
    # 将E矩阵强制成秩为2
    U, S, Vt = np.linalg.svd(E)
    S[2] = 0
    E = np.dot(U, np.dot(np.diag(S), Vt))
    return E


def decompose_essential_matrix(E):
    # 使用 SVD 对本质矩阵进行分解
    U, _, Vt = np.linalg.svd(E)
    # 创建一个 W 矩阵，用于计算旋转矩阵
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    # 计算两种可能的旋转矩阵
    R1 = np.dot(U, np.dot(W, Vt))
    R2 = np.dot(U, np.dot(W.T, Vt))
    # 确保 R1 和 R2 的行列式大于 0
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2
    # 计算两种可能的平移向量
    t1 = U[:, 2]
    t2 = -U[:, 2]
    return R_t_to_T(R1, t1), R_t_to_T(R2, t2), R_t_to_T(R1, t2), R_t_to_T(R2, t1)


def normal_equation(X, Y):
    """
    Solve θ in: X@θ = Y
    Where X's shape is (n, c), Y's shape is (n, 1) return θ(c, 1)
    Support solve in batch
    """
    if Y.shape[-1] != 1:
        Y = Y[:, None]
    if X.ndim == 2:
        return inv(X.T @ X) @ X.T @ Y
    # Batch computing
    if Y.ndim == 2:
        Y = np.array([Y] * len(X))
    XtX = np.einsum("bij,bjk->bik", X.transpose(0, 2, 1), X)
    XtY = np.einsum("bij,bjk->bik", X.transpose(0, 2, 1), Y)
    inv_XtX = np.linalg.inv(XtX)
    return np.einsum("bij,bjk->bik", inv_XtX, XtY)


def uvs_to_xyz_noramls(uvs, K):
    return np.pad(uvs, ((0, 0), (0, 1)), constant_values=1) @ inv(K).T


def matched_xyz_normals_to_zs(xyz_normals1, xyz_normals2, T_1to2):
    # X2*Z2 = R@X1*Z1 + t
    # [-R@X1, X2]@ [Z1, Z2] = t  --> X@theta = Y
    X = np.concatenate(
        ((-xyz_normals1 @ T_1to2[:3, :3].T)[:, :, None], xyz_normals2[:, :, None]), 2
    )
    Y = T_1to2[:3, 3:]
    thetas = normal_equation(X, Y)
    zs1, zs2 = np.squeeze(thetas).T
    return dict(zs1=zs1, zs2=zs2)


class EssentialMatrixStereo(Stereo):
    def __init__(
        self,
        uvs1,
        uvs2,
        K1,
        K2=None,
        baseline=1,
        xy1=None,
        xy2=None,
        name1="cam1",
        name2="cam2",
    ):
        if K2 is None:
            K2 = K1
        assert name1 != name2
        xyz_normals1 = uvs_to_xyz_noramls(uvs1, K1)
        xyz_normals2 = uvs_to_xyz_noramls(uvs2, K2)
        E = compute_essential_matrix(xyz_normals1, xyz_normals2)
        T_1to2s = decompose_essential_matrix(E)

        for T in T_1to2s:
            # z2_z1 = calibrating.apply_T_to_point_cloud(T, xyz_normals1)/xyz_normals2
            # print(z2_z1)

            T[:3, 3] *= baseline / np.linalg.norm(T[:3, 3])
            # TODO: Dose this need slice to recduce compuate?
            zs = matched_xyz_normals_to_zs(xyz_normals1, xyz_normals2, T)
            z1, z2 = zs["zs1"].mean(), zs["zs2"].mean()
            if z1 > 0 and z2 > 0:
                break
        if boxx.mg():
            print(z1, z2)
        xy1 = np.int32(K1[:2, 3]).tolist() if xy1 is None else xy1
        xy2 = np.int32(K2[:2, 3]).tolist() if xy2 is None else xy2
        super().__init__()
        self.R, self.t = T[:3, :3], T[:3, 3]
        self.load(
            dict(
                R=self.R,
                t=self.t,
                cam1=dict(xy=xy1, K=K1, name=str(name1)),
                cam2=dict(xy=xy2, K=K2, name=str(name2)),
            )
        )
        self.epipolar = zs
        self.epipolar["E"] = E
        self.epipolar["uvs1"] = uvs1
        self.epipolar["uvs2"] = uvs2
        self.epipolar["z1"] = z1
        self.epipolar["z2"] = z2

    @classmethod
    def from_stereo(cls, uvs1, uvs2, stereo, baseline=None):
        self = cls(
            uvs1,
            uvs2,
            K1=stereo.cam1.K,
            K2=stereo.cam2.K,
            xy1=stereo.cam1.xy,
            xy2=stereo.cam2.xy,
            baseline=baseline or stereo.baseline,
        )
        dic = stereo.dump(return_dict=1)
        dic["R"], dic["t"] = self.R, self.t
        self.load(dic)
        return self

    def align_scale_with(stereo1, stereo2, matched=None):
        names1 = stereo1.cam1.name, stereo1.cam2.name
        names2 = stereo2.cam1.name, stereo2.cam2.name
        assert len(set(names1)) == 2, names1
        assert len(set(names2)) == 2, names2
        assert (
            len(set(names1 + names2)) != 2
        ), f"names1={names1}, names2={names2} has exactly the same names!"
        assert (
            len(set(names1 + names2)) != 4
        ), f"names1={names1}, names2={names2} has total different names!"
        for name in names1:
            if name in names2:
                break
        s1_suffix = str(names1.index(name) + 1)
        s2_suffix = str(names2.index(name) + 1)
        if matched is None:
            uvs1 = stereo1.epipolar["uvs" + s1_suffix]
            uvs2 = stereo2.epipolar["uvs" + s2_suffix]
            matched = matching_uvs_in_one_img(uvs1, uvs2)
        assert len(matched["uv_match_idx1"]) > 10
        z_matched1 = stereo1.epipolar["zs" + s1_suffix][matched["uv_match_idx1"]]
        z_matched2 = stereo2.epipolar["zs" + s2_suffix][matched["uv_match_idx2"]]
        rate = z_matched2.mean() / z_matched1.mean()
        stereo1.set_scale(rate)

    def set_scale(self, rate):
        self.t = self.t * rate
        for key in [
            "z1",
            "z2",
            "zs1",
            "zs2",
        ]:
            self.epipolar[key] *= rate


def filter_overlap_uvs(uvs1, uvs2):
    _, inverse, count_ = np.unique(
        np.int32(uvs1.round()), axis=0, return_counts=True, return_inverse=True
    )
    count1 = count_[inverse]

    _, inverse, count_, = np.unique(
        np.int32(uvs2.round()), axis=0, return_counts=True, return_inverse=True
    )
    count2 = count_[inverse]
    mask = (count1 <= 1) & (count2 <= 1)
    return uvs1[mask], uvs2[mask]


def matching_uvs_in_one_img_precise(uvs1, uvs2, MAX_DISTANCE=1, MIN_MATCHED_PIXELS=10):
    from scipy.spatial import KDTree

    kd_tree = KDTree(uvs1)
    distance, nearst_idx = kd_tree.query(uvs2)

    mask = distance < MAX_DISTANCE

    k = int((uvs1.size + uvs2.size) / 2 * 0.1)

    if mask.sum() > MIN_MATCHED_PIXELS and mask.sum() > k:
        arg1 = np.argpartition(distance, k)[:k]
        mask_ = mask * False
        mask_[arg1] = mask[arg1]
        mask = mask_
    if mask.sum() < MIN_MATCHED_PIXELS:
        return {}
    return dict(
        uv_match_idx2=np.arange(0, mask.size)[mask], uv_match_idx1=nearst_idx[mask]
    )


def matching_uvs_in_one_img(
    uvs1, uvs2, MAX_DISTANCE=1, MIN_MATCHED_PIXELS=10, precise=False
):
    """
    time: 213.7871 ->  4.697554
    precise: Almost unchanged
    """
    if precise:
        return matching_uvs_in_one_img_precise(
            uvs1, uvs2, MAX_DISTANCE, MIN_MATCHED_PIXELS
        )
    cell1 = np.int32((uvs1 / MAX_DISTANCE).round())
    unique1, index1 = np.unique(cell1, return_index=True, axis=0)
    cell2 = np.int32((uvs2 / MAX_DISTANCE).round())
    unique2, index2 = np.unique(cell2, return_index=True, axis=0)

    dtype = unique1.dtype
    dtypes = np.dtype([(str(x), dtype) for x in enumerate(unique1[0])])

    intersect_, idx_unique1, idx_unique2 = np.intersect1d(
        unique1.view(dtypes),
        unique2.view(dtypes),
        return_indices=True,
        assume_unique=True,
    )
    if idx_unique1.size < MIN_MATCHED_PIXELS:
        return {}
    return dict(
        uv_match_idx1=index1[idx_unique1],
        uv_match_idx2=index2[idx_unique2],
    )


if __name__ == "__main__":
    from boxx import *
    from calibrating import get_test_cams, T_to_deg_distance, perturb_T

    cam1, cam2, cam3 = get_test_cams("aruco").values()
    key = list(cam1)[0]
    img1 = boxx.imread(cam1[key]["path"])
    img2 = boxx.imread(cam2[key]["path"])
    stereo = Stereo(cam1, cam2)
    # {"K1":stereo.K.tolist(),"K2":stereo.K.tolist(), "uvs1uvs2": np.random.randint(1,1024,(20,4)).tolist()}
    uvs1, uvs2, obj_points = stereo.get_conjoint_points()
    uvs1_distort = np.concatenate(uvs1, 0)
    uvs2_distort = np.concatenate(uvs2, 0)
    uvs1 = cam1.undistort_points(uvs1_distort)
    uvs2 = cam1.undistort_points(uvs2_distort)
    T_gt = R_t_to_T(stereo.R, stereo.t)

    random_test_case = 0  # + 1
    if random_test_case:
        T_gt = perturb_T(T_gt)
        T_boards = [d["T"] for d in cam1.values() if "T" in d]

        def imaging(K, T=np.eye(4), obj_points=None):
            if obj_points is None:
                obj_points = cam1.board.all_object_points
            uvzs = apply_T_to_point_cloud(T, obj_points) @ K.T
            return uvzs[:, :2] / uvzs[:, 2:]

        uvs1 = np.concatenate([imaging(cam1.K, T) for T in T_boards])
        uvs2 = np.concatenate([imaging(cam2.K, T_gt @ T) for T in T_boards])

    uvs1, uvs2 = filter_overlap_uvs(uvs1, uvs2)
    estereo = EssentialMatrixStereo.from_stereo(
        uvs1, uvs2, stereo, baseline=np.linalg.norm(T_gt[:3, 3])
    )
    T_re = estereo.T
    print("T_gt:  ", T_to_deg_distance(T_gt))
    print("T_re:  ", T_to_deg_distance(T_re))
    print("T_diff:", T_to_deg_distance(T_gt @ inv(T_re)))

    if not random_test_case:
        estereo.shows((*estereo.rectify(img1, img2)))

    """
    单位尺度: 主视图的平均深度为 1 个单位深度
    尺度传导: 一个视野, 分别和另外两个视野求了匹配的 uvs, 需要求出能被三个视野共同看见的 uvs, 用来传导尺度: 
        - 真实情况: 先 round, 做粗糙匹配, 互相匹配, 取 uv 距离最低的 20% 和 max(100) 来
        - stereo 关键点仿真数据: 由于单个像素会存在对应两张图片的关键点的情况, 所以需要去除这种同一 uv 坐标的多个关键点
    """
    #%%
    img3 = boxx.imread(cam3[key]["path"])
    stereo23 = Stereo(cam2, cam3)
    uvs3, uvs4, obj_points = stereo23.get_conjoint_points()
    uvs3 = cam1.undistort_points(np.concatenate(uvs3, 0))
    uvs4 = np.concatenate(uvs4, 0)
    uvs3, uvs4 = filter_overlap_uvs(uvs3, uvs4)
    estereo23 = EssentialMatrixStereo.from_stereo(uvs3, uvs4, stereo23, baseline=1)
    # stereo23.shows((*stereo23.rectify(img2, img3)))
    # estereo23.shows((*estereo23.rectify(img2, img3)))

    estereo23.align_scale_with(estereo)
    estereo23.align_scale_with(estereo)  # 验证重复 align
    #%%
    stereo13 = Stereo(cam1, cam3)
    stereo13_re = Stereo.load(dict(T=estereo23.T @ estereo.T, cam1=cam1, cam2=cam3))

    print(stereo13, stereo13_re)
    print(T_to_deg_distance(stereo13.T, stereo13_re.T))
    stereo13.shows((*stereo13.rectify(img1, img3)))
    stereo13_re.shows((*stereo13_re.rectify(img1, img3)))
