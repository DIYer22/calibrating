#!/usr/bin/env python3

import boxx
import numpy as np

with boxx.inpkg():
    from .camera import Cam
    from .stereo_camera import Stereo
    from .utils import inv, R_t_to_T, apply_T_to_point_cloud


def compute_essential_matrix(xyz1, xyz2):
    assert xyz1.shape == xyz2.shape, "The shapes of input points must be the same."
    n = xyz1.shape[0]
    assert n >= 8, "At least 8 point pairs are required."

    # 构建矩阵A
    A = np.zeros((n, 9))
    for i in range(n):
        x1, y1, z1 = xyz1[i]
        x2, y2, z2 = xyz2[i]
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
    # Batch
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
    # X = X[0]
    Y = T_1to2[:3, 3:]
    thetas = normal_equation(X, Y)
    zs1, zs2 = np.squeeze(thetas).T
    return dict(zs1=zs1, zs2=zs2)


class EssentialMatrixStereo(Stereo):
    def __init__(self, uv1, uv2, K1, K2=None, baseline=1, xy1=None, xy2=None):
        if K2 is None:
            K2 = K1
        xyz_normals1 = uvs_to_xyz_noramls(uv1, K1)
        xyz_normals2 = uvs_to_xyz_noramls(uv2, K2)
        self.E = compute_essential_matrix(xyz_normals1, xyz_normals2)
        T_1to2s = decompose_essential_matrix(self.E)

        for T in T_1to2s:
            # z2_z1 = calibrating.apply_T_to_point_cloud(T, xyz_normals1)/xyz_normals2
            # print(z2_z1)

            T[:3, 3] *= baseline / np.linalg.norm(T[:3, 3])
            re = matched_xyz_normals_to_zs(xyz_normals1, xyz_normals2, T)
            z1, z2 = re["zs1"].mean(), re["zs2"].mean()
            if z1 > 0 and z2 > 0:
                break
        if boxx.mg():
            print(z1, z2)
        xy1 = np.int32(K1[:2, 3]).tolist() if xy1 is None else xy1
        xy2 = np.int32(K2[:2, 3]).tolist() if xy2 is None else xy2
        super().__init__()
        self.R, self.t = T[:3, :3], T[:3, 3]
        self.load(
            dict(R=self.R, t=self.t, cam1=dict(xy=xy1, K=K1), cam2=dict(xy=xy2, K=K2))
        )

    @classmethod
    def from_stereo(cls, uv1, uv2, stereo, baseline=None):
        return cls(
            uv1,
            uv2,
            K1=stereo.cam1.K,
            K2=stereo.cam2.K,
            xy1=stereo.cam1.xy,
            xy2=stereo.cam2.xy,
            baseline=baseline or stereo.baseline,
        )


if __name__ == "__main__":
    import cv2
    from boxx import *
    from calibrating import get_test_cams, T_to_deg_distance, perturb_T

    cam1, cam2, camd = get_test_cams("aruco").values()
    key = list(cam1)[0]
    img1, img2 = (
        boxx.imread(cam1[key]["path"]),
        boxx.imread(cam2[key]["path"]),
    )
    # img1_undistort = cv2.undistort(img1, cam1.K, cam1.D)
    # img2_undistort = cv2.undistort(img2, cam2.K, cam2.D)
    stereo = Stereo(cam1, cam2)

    # {"K1":stereo.K.tolist(),"K2":stereo.K.tolist(), "uv1uv2": np.random.randint(1,1024,(20,4)).tolist()}
    uv1, uv2, obj = stereo.get_conjoint_points()
    uv1_distort = np.concatenate(uv1, 0)
    uv2_distort = np.concatenate(uv2, 0)
    uv1 = cam1.undistort_points(uv1_distort)
    uv2 = cam1.undistort_points(uv2_distort)
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

        uv1 = np.concatenate([imaging(cam1.K, T) for T in T_boards])
        uv2 = np.concatenate([imaging(cam2.K, T_gt @ T) for T in T_boards])

    stereo2 = EssentialMatrixStereo.from_stereo(
        uv1, uv2, stereo, baseline=np.linalg.norm(T_gt[:3, 3])
    )
    stereo2 = Stereo.load(dict(T=stereo2.T, cam1=cam1, cam2=cam2))
    T_re = stereo2.T
    print("T_gt:  ", T_to_deg_distance(T_gt))
    print("T_re:  ", T_to_deg_distance(T_re))
    print("T_diff:", T_to_deg_distance(T_gt @ inv(T_re)))

    if not random_test_case:
        stereo2.shows((*stereo2.rectify(img1, img2)))
