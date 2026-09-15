import cv2
import numpy as np
from dataclasses import dataclass
import pycolmap
from .transformation_np import *


@dataclass
class CameraParams:
    width: int
    height: int
    focal_length: float = None  # Use sqrt(width^2 + height^2) if not provided FOV~=53°
    cx: float = None  # Use half of width if not provided
    cy: float = None  # Use half of height if not provided


class Cv2RansacEssentialSolver:
    def __init__(self, camera_params: CameraParams):
        width = camera_params.width
        height = camera_params.height
        focal_length = camera_params.focal_length
        if focal_length is None:
            focal_length = (width**2 + height**2) ** 0.5
        cx = camera_params.cx
        cy = camera_params.cy
        if cx is None:
            cx = width / 2
        if cy is None:
            cy = height / 2

        self.camera_matrix = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]])

    def get_K(self):
        """
        Returns:
            K: np.ndarray, shape (3, 3), dtype=np.float32
        """
        return self.camera_matrix

    def solve(self, pts0, pts1):
        # Find essential matrix with stricter RANSAC
        E, mask = cv2.findEssentialMat(
            pts0,
            pts1,
            self.camera_matrix,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )

        # Recover pose
        _, R, t, mask = cv2.recoverPose(E, pts0, pts1, self.camera_matrix, mask=mask)

        return R, t


class PycolmapRansacTwoViewGeometrySolver:
    def __init__(self, camera_params: CameraParams):
        width = camera_params.width
        height = camera_params.height
        focal_length = camera_params.focal_length
        if focal_length is None:
            focal_length = (width**2 + height**2) ** 0.5
        cx = camera_params.cx
        cy = camera_params.cy
        if cx is None:
            cx = width / 2
        if cy is None:
            cy = height / 2
        self.camera_matrix = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]])

        # Set up pycolmap
        self.camera = pycolmap.Camera(
            camera_id=0,
            model="SIMPLE_PINHOLE",
            width=width,
            height=height,
            params=[focal_length, cx, cy],
        )

        # Configure options for consecutive frames
        self.options = pycolmap.TwoViewGeometryOptions(
            min_num_inliers=10,
            min_E_F_inlier_ratio=0.8,
            max_H_inlier_ratio=0.9,
            compute_relative_pose=True,
        )
        print(self.options.summary())

    def get_K(self):
        return self.camera_matrix

    def solve(self, pts0, pts1):
        min_matches = int(getattr(self.options, "min_num_inliers", 10))
        if len(pts0) < min_matches or len(pts1) < min_matches:
            print("[SimpleVO] Warning: not enough matches; reusing previous camera pose.")
            return np.eye(4, dtype=np.float32)

        matches = np.stack([np.arange(len(pts0)), np.arange(len(pts0))], axis=-1)
        answer = pycolmap.estimate_calibrated_two_view_geometry(
            self.camera,
            pts0.astype(np.float64),
            self.camera,
            pts1.astype(np.float64),
            matches=matches,
            options=self.options,
        )
        cam2_from_cam1 = None if answer is None else getattr(answer, "cam2_from_cam1", None)
        if cam2_from_cam1 is None:
            print("[SimpleVO] Warning: two-view geometry failed; reusing previous camera pose.")
            return np.eye(4, dtype=np.float32)

        # cam2_from_cam1 means T_0_to_1 in our language
        Rt = cam2_from_cam1.matrix().astype(np.float32)  # shape (3, 4)
        T = np.eye(4, dtype=np.float32)
        T[:3] = Rt
        return T


two_pair_solver_map = {
    # "cv2": Cv2RansacEssentialSolver,  # This is not stable
    "pycolmap": PycolmapRansacTwoViewGeometrySolver,  # Essential and Homography at the same time
}


class TwoPairSolver:
    def __init__(self, params: CameraParams, solver: str = "pycolmap"):
        self.solver = two_pair_solver_map[solver](params)
        self._params = params
        self._solver_name = solver

    def clone(self):
        return TwoPairSolver(self._params, solver=self._solver_name)

    def get_K(self):
        """
        Returns:
            K: np.ndarray, shape (3, 3), dtype=np.float32
        """
        return self.solver.get_K()

    def solve(self, pts0, pts1):
        """
        Args:
            pts0: np.ndarray, shape (N, 2), dtype=np.float32
            pts1: np.ndarray, shape (N, 2), dtype=np.float32
        Returns:
            T: np.ndarray, shape (4, 4), dtype=np.float32
        """
        return self.solver.solve(pts0, pts1)


########################################################
# Interpolate missing frames
########################################################


def interpolate_missing_frames(T_w2c_list, sample_idxs):
    """
    对给定的 T_w2c_list（已知帧的变换矩阵）进行平滑插值，生成所有帧的变换矩阵。
    其中：
      - 平移部分采用线性插值；
      - 旋转部分采用自实现的SLERP球面线性插值，保证旋转过渡平滑。

    参数：
        T_w2c_list (numpy.ndarray): 形状为 (F, 4, 4) 的已知变换矩阵数组
        sample_idxs (list 或 numpy.ndarray): 长度为 F 的已知帧在原始序列中的索引
          （假设第一个索引为 0，最后一个为 F_all - 1）

    返回：
        numpy.ndarray: 形状为 (F_all, 4, 4) 的所有帧的变换矩阵，缺失帧通过平滑插值填充。
    """
    sample_idxs = np.array(sample_idxs)
    # 根据最后一个已知帧索引确定总帧数（假设索引从 0 开始）
    F_all = sample_idxs[-1] + 1
    new_T_list = []

    # 分离出平移和旋转部分
    translations = np.array([T[:3, 3] for T in T_w2c_list])
    rotations = np.array([T[:3, :3] for T in T_w2c_list])
    # 将旋转矩阵转换为四元数
    quaternions = np.array([rotation_matrix_to_quaternion(R) for R in rotations])

    for i in range(F_all):
        # 如果该帧为已知帧，直接使用对应的变换矩阵
        if i in sample_idxs:
            known_index = np.where(sample_idxs == i)[0][0]
            new_T_list.append(T_w2c_list[known_index])
        else:
            # 定位左右两侧已知帧
            next_known = np.searchsorted(sample_idxs, i)
            prev_known = next_known - 1
            # 计算插值比例 t
            t_interp = (i - sample_idxs[prev_known]) / (sample_idxs[next_known] - sample_idxs[prev_known])
            # 平移部分：线性插值
            trans_interp = (1 - t_interp) * translations[prev_known] + t_interp * translations[next_known]
            # 旋转部分：自实现 SLERP 插值
            q0 = quaternions[prev_known]
            q1 = quaternions[next_known]
            q_interp = slerp(q0, q1, t_interp)
            rot_interp = quaternion_to_rotation_matrix(q_interp)
            # 构造最终的 4x4 变换矩阵
            T_interp = np.eye(4)
            T_interp[:3, :3] = rot_interp
            T_interp[:3, 3] = trans_interp
            new_T_list.append(T_interp)

    return np.array(new_T_list)
