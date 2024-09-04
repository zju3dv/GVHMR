import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pytorch3d.transforms import so3_exp_map, so3_log_map
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_axis_angle, matrix_to_rotation_6d
import pytorch3d.ops.knn as knn
from hmr4d.utils.pylogger import Log
from pytorch3d.transforms import euler_angles_to_matrix
import hmr4d.utils.matrix as matrix
from einops import einsum, rearrange, repeat
from hmr4d.utils.geo.quaternion import qbetween


def homo_points(points):
    """
    Args:
        points: (..., C)
    Returns: (..., C+1), with 1 padded
    """
    return F.pad(points, [0, 1], value=1.0)


def apply_Ts_on_seq_points(points, Ts):
    """
    perform translation matrix on related point
    Args:
        points: (..., N, 3)
        Ts: (..., N, 4, 4)
    Returns: (..., N, 3)
    """
    points = torch.torch.einsum("...ki,...i->...k", Ts[..., :3, :3], points) + Ts[..., :3, 3]
    return points


def apply_T_on_points(points, T):
    """
    Args:
        points: (..., N, 3)
        T: (..., 4, 4)
    Returns: (..., N, 3)
    """
    points_T = torch.einsum("...ki,...ji->...jk", T[..., :3, :3], points) + T[..., None, :3, 3]
    return points_T


def T_transforms_points(T, points, pattern):
    """manual mode of apply_T_on_points
    T: (..., 4, 4)
    points: (..., 3)
    pattern: "... c d, ... d -> ... c"
    """
    return einsum(T, homo_points(points), pattern)[..., :3]


def project_p2d(points, K=None, is_pinhole=True):
    """
    Args:
        points: (..., (N), 3)
        K: (..., 3, 3)
    Returns: shape is similar to points but without z
    """
    points = points.clone()
    if is_pinhole:
        z = points[..., [-1]]
        z.masked_fill_(z.abs() < 1e-6, 1e-6)
        points_proj = points / z
    else:  # orthogonal
        points_proj = F.pad(points[..., :2], (0, 1), value=1)

    if K is not None:
        # Handle N
        if len(points_proj.shape) == len(K.shape):
            p2d_h = torch.einsum("...ki,...ji->...jk", K, points_proj)
        else:
            p2d_h = torch.einsum("...ki,...i->...k", K, points_proj)
    else:
        p2d_h = points_proj[..., :2]

    return p2d_h[..., :2]


def gen_uv_from_HW(H, W, device="cpu"):
    """Returns: (H, W, 2), as float. Note: uv not ij"""
    grid_v, grid_u = torch.meshgrid(torch.arange(H), torch.arange(W))
    return (
        torch.stack(
            [grid_u, grid_v],
            dim=-1,
        )
        .float()
        .to(device)
    )  # (H, W, 2)


def unproject_p2d(uv, z, K):
    """we assume a pinhole camera for unprojection
    uv: (B, N, 2)
    z: (B, N, 1)
    K: (B, 3, 3)
    Returns: (B, N, 3)
    """
    xy_atz1 = (uv - K[:, None, :2, 2]) / K[:, None, [0, 1], [0, 1]]  # (B, N, 2)
    xyz = torch.cat([xy_atz1 * z, z], dim=-1)
    return xyz


def cvt_p2d_from_i_to_c(uv, K):
    """
    Args:
        uv: (..., 2) or (..., N, 2)
        K: (..., 3, 3)
    Returns: the same shape as input uv
    """
    if len(uv.shape) == len(K.shape):
        xy = (uv - K[..., None, :2, 2]) / K[..., None, [0, 1], [0, 1]]
    else:  # without N
        xy = (uv - K[..., :2, 2]) / K[..., [0, 1], [0, 1]]
    return xy


def cvt_to_bi01_p2d(p2d, bbx_lurb):
    """
    p2d: (..., (N), 2)
    bbx_lurb: (..., 4)
    """
    if len(p2d.shape) == len(bbx_lurb.shape) + 1:
        bbx_lurb = bbx_lurb[..., None, :]

    bbx_wh = bbx_lurb[..., 2:] - bbx_lurb[..., :2]
    bi01_p2d = (p2d - bbx_lurb[..., :2]) / bbx_wh
    return bi01_p2d


def cvt_from_bi01_p2d(bi01_p2d, bbx_lurb):
    """Use bbx_lurb to resize bi01_p2d to p2d (image-coordinates)
    Args:
        p2d: (..., 2) or (..., N, 2)
        bbx_lurb: (..., 4)
    Returns:
        p2d: shape is the same as input
    """
    bbx_wh = bbx_lurb[..., 2:] - bbx_lurb[..., :2]  # (..., 2)
    if len(bi01_p2d.shape) == len(bbx_wh.shape) + 1:
        p2d = (bi01_p2d * bbx_wh.unsqueeze(-2)) + bbx_lurb[..., None, :2]
    else:
        p2d = (bi01_p2d * bbx_wh) + bbx_lurb[..., :2]
    return p2d


def cvt_p2d_from_bi01_to_c(bi01, bbxs_lurb, Ks):
    """
    Args:
        bi01: (..., (N), 2), value in range (0,1), the point in the bbx image
        bbxs_lurb: (..., 4)
        Ks: (..., 3, 3)
    Returns:
        c: (..., (N), 2)
    """
    i = cvt_from_bi01_p2d(bi01, bbxs_lurb)
    c = cvt_p2d_from_i_to_c(i, Ks)
    return c


def cvt_p2d_from_pm1_to_i(p2d_pm1, bbx_xys):
    """
    Args:
        p2d_pm1: (..., (N), 2), value in range (-1,1), the point in the bbx image
        bbx_xys: (..., 3)
    Returns:
        p2d: (..., (N), 2)
    """
    return bbx_xys[..., :2] + p2d_pm1 * bbx_xys[..., [2]] / 2


def uv2l_index(uv, W):
    return uv[..., 0] + uv[..., 1] * W


def l2uv_index(l, W):
    v = torch.div(l, W, rounding_mode="floor")
    u = l % W
    return torch.stack([u, v], dim=-1)


def transform_mat(R, t):
    """
    Args:
        R: Bx3x3 array of a batch of rotation matrices
        t: Bx3x(1) array of a batch of translation vectors
    Returns:
        T: Bx4x4 Transformation matrix
    """
    # No padding left or right, only add an extra row
    if len(R.shape) > len(t.shape):
        t = t[..., None]
    return torch.cat([F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=-1)


def axis_angle_to_matrix_exp_map(aa):
    """use pytorch3d so3_exp_map
    Args:
        aa: (*, 3)
    Returns:
        R: (*, 3, 3)
    """
    print("Use pytorch3d.transforms.axis_angle_to_matrix instead!!!")
    ori_shape = aa.shape[:-1]
    return so3_exp_map(aa.reshape(-1, 3)).reshape(*ori_shape, 3, 3)


def matrix_to_axis_angle_log_map(R):
    """use pytorch3d so3_log_map
    Args:
        aa: (*, 3, 3)
    Returns:
        R: (*, 3)
    """
    print("WARINING! I met singularity problem with this function, use matrix_to_axis_angle instead!")
    ori_shape = R.shape[:-2]
    return so3_log_map(R.reshape(-1, 3, 3)).reshape(*ori_shape, 3)


def matrix_to_axis_angle(R):
    """use pytorch3d so3_log_map
    Args:
        aa: (*, 3, 3)
    Returns:
        R: (*, 3)
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(R))


def ransac_PnP(K, pts_2d, pts_3d, err_thr=10):
    """solve pnp"""
    dist_coeffs = np.zeros(shape=[8, 1], dtype="float64")

    pts_2d = np.ascontiguousarray(pts_2d.astype(np.float64))
    pts_3d = np.ascontiguousarray(pts_3d.astype(np.float64))
    K = K.astype(np.float64)

    try:
        _, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d, K, dist_coeffs, reprojectionError=err_thr, iterationsCount=10000, flags=cv2.SOLVEPNP_EPNP
        )

        rotation = cv2.Rodrigues(rvec)[0]

        pose = np.concatenate([rotation, tvec], axis=-1)
        pose_homo = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)

        inliers = [] if inliers is None else inliers

        return pose, pose_homo, inliers
    except cv2.error:
        print("CV ERROR")
        return np.eye(4)[:3], np.eye(4), []


def ransac_PnP_batch(K_raw, pts_2d, pts_3d, err_thr=10):
    fit_R, fit_t = [], []
    for b in range(K_raw.shape[0]):
        pose, _, inliers = ransac_PnP(K_raw[b], pts_2d[b], pts_3d[b], err_thr=err_thr)
        fit_R.append(pose[:3, :3])
        fit_t.append(pose[:3, 3])
    fit_R = np.stack(fit_R, axis=0)
    fit_t = np.stack(fit_t, axis=0)
    return fit_R, fit_t


def triangulate_point(Ts_w2c, c_p2d, **kwargs):
    from hmr4d.utils.geo.triangulation import triangulate_persp

    print("Deprecated, please import from hmr4d.utils.geo.triangulation")
    return triangulate_persp(Ts_w2c, c_p2d, **kwargs)


def triangulate_point_ortho(Ts_w2c, c_p2d, **kwargs):
    from hmr4d.utils.geo.triangulation import triangulate_ortho

    print("Deprecated, please import from hmr4d.utils.geo.triangulation")
    return triangulate_ortho(Ts_w2c, c_p2d, **kwargs)


def get_nearby_points(points, query_verts, padding=0.0, p=1):
    """
    points: (S, 3)
    query_verts: (V, 3)
    """
    if p == 1:
        max_xyz = query_verts.max(0)[0] + padding
        min_xyz = query_verts.min(0)[0] - padding
        idx = (((points - min_xyz) > 0).all(dim=-1) * ((points - max_xyz) < 0).all(dim=-1)).nonzero().squeeze(-1)
        nearby_points = points[idx]
    elif p == 2:
        squared_dist, _, _ = knn.knn_points(points[None], query_verts[None], K=1, return_nn=False)
        mask = squared_dist[0, :, 0] < padding**2  # (S,)
        nearby_points = points[mask]

    return nearby_points


def unproj_bbx_to_fst(bbx_lurb, K, near_z=0.5, far_z=12.5):
    B = bbx_lurb.size(0)
    uv = bbx_lurb[:, [[0, 1], [2, 1], [2, 3], [0, 3], [0, 1], [2, 1], [2, 3], [0, 3]]]
    if isinstance(near_z, float):
        z = uv.new([near_z] * 4 + [far_z] * 4).reshape(1, 8, 1).repeat(B, 1, 1)
    else:
        z = torch.cat([near_z[:, None, None].repeat(1, 4, 1), far_z[:, None, None].repeat(1, 4, 1)], dim=1)
    c_frustum_points = unproject_p2d(uv, z, K)  # (B, 8, 3)
    return c_frustum_points


def convert_bbx_xys_to_lurb(bbx_xys):
    """
    Args: bbx_xys (..., 3) -> bbx_lurb (..., 4)
    """
    size = bbx_xys[..., 2:]
    center = bbx_xys[..., :2]
    lurb = torch.cat([center - size / 2, center + size / 2], dim=-1)
    return lurb


def convert_lurb_to_bbx_xys(bbx_lurb):
    """
    Args: bbx_lurb (..., 4) -> bbx_xys (..., 3) be aware that it is squared
    """
    size = (bbx_lurb[..., 2:] - bbx_lurb[..., :2]).max(-1, keepdim=True)[0]
    center = (bbx_lurb[..., :2] + bbx_lurb[..., 2:]) / 2
    return torch.cat([center, size], dim=-1)


# ================== AZ/AY Transformations ================== #


def compute_T_ayf2az(joints, inverse=False):
    """
    Args:
        joints: (B, J, 3), in the start-frame, az-coordinate
    Returns:
        if inverse == False:
           T_af2az: (B, 4, 4)
        else :
            T_az2af: (B, 4, 4)
    """

    t_ayf2az = joints[:, 0, :].detach().clone()
    t_ayf2az[:, 2] = 0  # do not modify z

    RL_xy_h = joints[:, 1, [0, 1]] - joints[:, 2, [0, 1]]  # (B, 2), hip point to left side
    RL_xy_s = joints[:, 16, [0, 1]] - joints[:, 17, [0, 1]]  # (B, 2), shoulder point to left side
    RL_xy = RL_xy_h + RL_xy_s
    I_mask = RL_xy.pow(2).sum(-1) < 1e-4  # do not rotate, when can't decided the face direction
    if I_mask.sum() > 0:
        Log.warn("{} samples can't decide the face direction".format(I_mask.sum()))
    x_dir = F.pad(F.normalize(RL_xy, 2, -1), (0, 1), value=0)  # (B, 3)
    y_dir = torch.zeros_like(x_dir)
    y_dir[..., 2] = 1
    z_dir = torch.cross(x_dir, y_dir, dim=-1)
    R_ayf2az = torch.stack([x_dir, y_dir, z_dir], dim=-1)  # (B, 3, 3)
    R_ayf2az[I_mask] = torch.eye(3).to(R_ayf2az)

    if inverse:
        R_az2ayf = R_ayf2az.transpose(1, 2)  # (B, 3, 3)
        t_az2ayf = -einsum(R_ayf2az, t_ayf2az, "b i j , b i -> b j")  # (B, 3)
        return transform_mat(R_az2ayf, t_az2ayf)
    else:
        return transform_mat(R_ayf2az, t_ayf2az)


def compute_T_ayfz2ay(joints, inverse=False):
    """
    Args:
        joints: (B, J, 3), in the start-frame, ay-coordinate
    Returns:
        if inverse == False:
            T_ayfz2ay: (B, 4, 4)
        else :
            T_ay2ayfz: (B, 4, 4)
    """
    t_ayfz2ay = joints[:, 0, :].detach().clone()
    t_ayfz2ay[:, 1] = 0  # do not modify y

    RL_xz_h = joints[:, 1, [0, 2]] - joints[:, 2, [0, 2]]  # (B, 2), hip point to left side
    RL_xz_s = joints[:, 16, [0, 2]] - joints[:, 17, [0, 2]]  # (B, 2), shoulder point to left side
    RL_xz = RL_xz_h + RL_xz_s
    I_mask = RL_xz.pow(2).sum(-1) < 1e-4  # do not rotate, when can't decided the face direction
    if I_mask.sum() > 0:
        Log.warn("{} samples can't decide the face direction".format(I_mask.sum()))

    x_dir = torch.zeros_like(t_ayfz2ay)  # (B, 3)
    x_dir[:, [0, 2]] = F.normalize(RL_xz, 2, -1)
    y_dir = torch.zeros_like(x_dir)
    y_dir[..., 1] = 1  # (B, 3)
    z_dir = torch.cross(x_dir, y_dir, dim=-1)
    R_ayfz2ay = torch.stack([x_dir, y_dir, z_dir], dim=-1)  # (B, 3, 3)
    R_ayfz2ay[I_mask] = torch.eye(3).to(R_ayfz2ay)

    if inverse:
        R_ay2ayfz = R_ayfz2ay.transpose(1, 2)
        t_ay2ayfz = -einsum(R_ayfz2ay, t_ayfz2ay, "b i j , b i -> b j")
        return transform_mat(R_ay2ayfz, t_ay2ayfz)
    else:
        return transform_mat(R_ayfz2ay, t_ayfz2ay)


def compute_T_ay2ayrot(joints):
    """
    Args:
        joints: (B, J, 3), in the start-frame, ay-coordinate
    Returns:
        T_ay2ayrot: (B, 4, 4)
    """
    t_ayrot2ay = joints[:, 0, :].detach().clone()
    t_ayrot2ay[:, 1] = 0  # do not modify y

    B = joints.shape[0]
    euler_angle = torch.zeros((B, 3), device=joints.device)
    yrot_angle = torch.rand((B,), device=joints.device) * 2 * torch.pi
    euler_angle[:, 0] = yrot_angle
    R_ay2ayrot = euler_angles_to_matrix(euler_angle, "YXZ")  # (B, 3, 3)

    R_ayrot2ay = R_ay2ayrot.transpose(1, 2)
    t_ay2ayrot = -einsum(R_ayrot2ay, t_ayrot2ay, "b i j , b i -> b j")
    return transform_mat(R_ay2ayrot, t_ay2ayrot)


def compute_root_quaternion_ay(joints):
    """
    Args:
        joints: (B, J, 3), in the start-frame, ay-coordinate
    Returns:
        root_quat: (B, 4) from z-axis to fz
    """
    joints_shape = joints.shape
    joints = joints.reshape((-1,) + joints_shape[-2:])
    t_ayfz2ay = joints[:, 0, :].detach().clone()
    t_ayfz2ay[:, 1] = 0  # do not modify y

    RL_xz_h = joints[:, 1, [0, 2]] - joints[:, 2, [0, 2]]  # (B, 2), hip point to left side
    RL_xz_s = joints[:, 16, [0, 2]] - joints[:, 17, [0, 2]]  # (B, 2), shoulder point to left side
    RL_xz = RL_xz_h + RL_xz_s
    I_mask = RL_xz.pow(2).sum(-1) < 1e-4  # do not rotate, when can't decided the face direction
    if I_mask.sum() > 0:
        Log.warn("{} samples can't decide the face direction".format(I_mask.sum()))

    x_dir = torch.zeros_like(t_ayfz2ay)  # (B, 3)
    x_dir[:, [0, 2]] = F.normalize(RL_xz, 2, -1)
    y_dir = torch.zeros_like(x_dir)
    y_dir[..., 1] = 1  # (B, 3)
    z_dir = torch.cross(x_dir, y_dir, dim=-1)

    z_dir[..., 2] += 1e-9
    pos_z_vec = torch.tensor([0, 0, 1]).to(joints.device).float()  # (3,)
    root_quat = qbetween(pos_z_vec[None], z_dir)  # (B, 4)
    root_quat = root_quat.reshape(joints_shape[:-2] + (4,))
    return root_quat


# ================== Transformations between two sets of features ================== #


def similarity_transform_batch(S1, S2):
    """
    Computes a similarity transform (sR, t) that solves the orthogonal Procrutes problem.
    Args:
        S1, S2: (*, L, 3)
    """
    assert S1.shape == S2.shape
    S_shape = S1.shape
    S1 = S1.reshape(-1, *S_shape[-2:])
    S2 = S2.reshape(-1, *S_shape[-2:])

    S1 = S1.transpose(-2, -1)
    S2 = S2.transpose(-2, -1)

    # --- The code is borrowed from WHAM ---
    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)  # axis is along N, S1(B, 3, N)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0, 2, 1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # -------
    # reshape back
    # sR = scale[:, None, None] * R
    # sR = sR.reshape(*S_shape[:-2], 3, 3)
    scale = scale.reshape(*S_shape[:-2], 1, 1)
    R = R.reshape(*S_shape[:-2], 3, 3)
    t = t.reshape(*S_shape[:-2], 3, 1)

    return (scale, R), t


def kabsch_algorithm_batch(X1, X2):
    """
    Computes a rigid transform (R, t)
    Args:
        X1, X2: (*, L, 3)
    """
    assert X1.shape == X2.shape
    X_shape = X1.shape
    X1 = X1.reshape(-1, *X_shape[-2:])
    X2 = X2.reshape(-1, *X_shape[-2:])

    # 1. 计算质心
    centroid_X1 = torch.mean(X1, dim=-2, keepdim=True)
    centroid_X2 = torch.mean(X2, dim=-2, keepdim=True)

    # 2. 去中心化
    X1_centered = X1 - centroid_X1
    X2_centered = X2 - centroid_X2

    # 3. 计算协方差矩阵
    H = torch.matmul(X1_centered.transpose(-2, -1), X2_centered)

    # 4. 奇异值分解
    U, S, Vt = torch.linalg.svd(H)

    # 5. 计算旋转矩阵
    R = torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1))

    # 修正反射矩阵
    d = (torch.det(R) < 0).unsqueeze(-1).unsqueeze(-1)
    Vt = torch.where(d, -Vt, Vt)
    R = torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1))

    # 6. 计算平移向量
    t = centroid_X2.transpose(-2, -1) - torch.matmul(R, centroid_X1.transpose(-2, -1))

    # -------
    # reshape back
    R = R.reshape(*X_shape[:-2], 3, 3)
    t = t.reshape(*X_shape[:-2], 3, 1)

    return R, t


# ===== WHAM cam_angvel ===== #


def compute_cam_angvel(R_w2c, padding_last=True):
    """
    R_w2c : (F, 3, 3)
    """
    # R @ R0 = R1, so R = R1 @ R0^T
    cam_angvel = matrix_to_rotation_6d(R_w2c[1:] @ R_w2c[:-1].transpose(-1, -2))  # (F-1, 6)
    # cam_angvel = (cam_angvel - torch.tensor([[1, 0, 0, 0, 1, 0]])) * FPS
    assert padding_last
    cam_angvel = torch.cat([cam_angvel, cam_angvel[-1:]], dim=0)  # (F, 6)
    return cam_angvel.float()


def ransac_gravity_vec(xyz, num_iterations=100, threshold=0.05, verbose=False):
    # xyz: (L, 3)
    N = xyz.shape[0]
    max_inliers = []
    best_model = None
    norms = xyz.norm(dim=-1)  # (L,)

    for _ in range(num_iterations):
        # 随机选择一个样本
        sample_index = np.random.randint(N)
        sample = xyz[sample_index]  # (3,)

        # 计算所有点与样本点的角度差
        dot_product = (xyz * sample).sum(dim=-1)  # (L,)
        angles = dot_product / norms * norms[sample_index]  # (L,)
        angles = torch.clamp(angles, -1, 1)  # 防止数值误差导致的异常
        angles = torch.acos(angles)

        # 确定内点
        inliers = xyz[angles < threshold]

        if len(inliers) > len(max_inliers):
            max_inliers = inliers
            best_model = sample
        if len(max_inliers) == N:
            break
    if verbose:
        print(f"Inliers: {len(max_inliers)} / {N}")
    result = max_inliers.mean(dim=0)

    return result, max_inliers


def sequence_best_cammat(w_j3d, c_j3d, cam_rot):
    # get best camera estimation along the sequence, requires static camera
    # w_j3d: (L, J, 3)
    # c_j3d: (L, J, 3)
    # cam_rot: (L, 3, 3)

    L, J, _ = w_j3d.shape

    root_in_w = w_j3d[:, 0]  # (L, 3)
    root_in_c = c_j3d[:, 0]  # (L, 3)
    cam_mat = matrix.get_TRS(cam_rot, root_in_w)  # (L, 4, 4)
    cam_pos = matrix.get_position_from(-root_in_c[:, None], cam_mat)[:, 0]  # (L, 3)
    cam_mat = matrix.set_position(cam_mat, cam_pos)  # (L, 4, 4)

    w_j3d_expand = w_j3d[None].expand(L, -1, -1, -1)  # (L, L, J, 3)
    w_j3d_expand = w_j3d_expand.reshape(L, -1, 3)  # (L, L*J, 3)

    # get reproject error
    w_j3d_expand_in_c = matrix.get_relative_position_to(w_j3d_expand, cam_mat)  # (L, L*J, 3)
    w_j2d_expand_in_c = project_p2d(w_j3d_expand_in_c)  # (L, L*J, 2)
    w_j2d_expand_in_c = w_j2d_expand_in_c.reshape(L, L, J, 2)  # (L, L, J, 2)
    c_j2d = project_p2d(c_j3d)  # (L, J, 2)
    error = w_j2d_expand_in_c - c_j2d[None]  # (L, L, J, 2)
    error = error.norm(dim=-1).mean(dim=-1)  # (L, L)
    error = error.mean(dim=-1)  # (L,)
    ind = error.argmin()
    return cam_mat[ind], ind


def get_sequence_cammat(w_j3d, c_j3d, cam_rot):
    # w_j3d: (L, J, 3)
    # c_j3d: (L, J, 3)
    # cam_rot: (L, 3, 3)

    L, J, _ = w_j3d.shape

    root_in_w = w_j3d[:, 0]  # (L, 3)
    root_in_c = c_j3d[:, 0]  # (L, 3)
    cam_mat = matrix.get_TRS(cam_rot, root_in_w)  # (L, 4, 4)
    cam_pos = matrix.get_position_from(-root_in_c[:, None], cam_mat)[:, 0]  # (L, 3)
    cam_mat = matrix.set_position(cam_mat, cam_pos)  # (L, 4, 4)
    return cam_mat


def ransac_vec(vel, min_multiply=20, verbose=False):
    # xyz: (L, 3)
    # remove outlier velocity
    N = vel.shape[0]
    vel_1 = vel[None].expand(N, -1, -1)  # (L, L, 3)
    vel_2 = vel[:, None].expand(-1, N, -1)  # (L, L, 3)
    dist_mat = (vel_1 - vel_2).norm(dim=-1)  # (L, L)
    big_identity = torch.eye(N, device=vel.device) * 1e6
    dist_mat_ = dist_mat + big_identity
    threshold = dist_mat_.min() * min_multiply
    inner_mask = dist_mat < threshold  # (L, L)
    inner_num = inner_mask.sum(dim=-1)  # (L, )
    ind = inner_num.argmax()
    result = vel[inner_mask[ind]].mean(dim=0)  # (3,)
    if verbose:
        print(inner_mask[ind].sum().item())

    return result, inner_mask[ind]
