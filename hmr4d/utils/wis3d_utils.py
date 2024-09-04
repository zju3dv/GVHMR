from wis3d import Wis3D
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
from einops import einsum
from pytorch3d.transforms import axis_angle_to_matrix


def make_wis3d(output_dir="outputs/wis3d", name="debug", time_postfix=False):
    """
    Make a Wis3D instance. e.g.:
        from hmr4d.utils.wis3d_utils import make_wis3d
        wis3d = make_wis3d(time_postfix=True)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if time_postfix:
        time_str = datetime.now().strftime("%m%d-%H%M-%S")
        name = f"{name}_{time_str}"
        print(f"Creating Wis3D {name}")
    wis3d = Wis3D(output_dir.absolute(), name)
    return wis3d


color_schemes = {
    "red": ([255, 168, 154], [153, 17, 1]),
    "green": ([183, 255, 191], [0, 171, 8]),
    "blue": ([183, 255, 255], [0, 0, 255]),
    "cyan": ([183, 255, 255], [0, 255, 255]),
    "magenta": ([255, 183, 255], [255, 0, 255]),
    "black": ([0, 0, 0], [0, 0, 0]),
    "orange": ([255, 183, 0], [255, 128, 0]),
    "grey": ([203, 203, 203], [203, 203, 203]),
}


def get_gradient_colors(scheme="red", num_points=120, alpha=1.0):
    """
    Return a list of colors that are gradient from start to end.
    """
    start_rgba = torch.tensor(color_schemes[scheme][0] + [255 * alpha]) / 255
    end_rgba = torch.tensor(color_schemes[scheme][1] + [255 * alpha]) / 255
    colors = torch.stack([torch.linspace(s, e, steps=num_points) for s, e in zip(start_rgba, end_rgba)], dim=-1)
    return colors


def get_const_colors(name="red", partial_shape=(120, 5), alpha=1.0):
    """
    Return colors (partial_shape, 4)
    """
    rgba = torch.tensor(color_schemes[name][1] + [255 * alpha]) / 255
    partial_shape = tuple(partial_shape)
    colors = rgba[None].repeat(*partial_shape, 1)
    return colors


def get_colors_by_conf(conf, low="red", high="green"):
    colors = torch.stack([conf] * 3, dim=-1)
    colors = colors * torch.tensor(color_schemes[high][1]) + (1 - colors) * torch.tensor(color_schemes[low][1])
    return colors


# ================== Colored Motion Sequence ================== #


KINEMATIC_CHAINS = {
    "smpl22": [
        [0, 2, 5, 8, 11],  # right-leg
        [0, 1, 4, 7, 10],  # left-leg
        [0, 3, 6, 9, 12, 15],  # spine
        [9, 14, 17, 19, 21],  # right-arm
        [9, 13, 16, 18, 20],  # left-arm
    ],
    "h36m17": [
        [0, 1, 2, 3],  # right-leg
        [0, 4, 5, 6],  # left-leg
        [0, 7, 8, 9, 10],  # spine
        [8, 14, 15, 16],  # right-arm
        [8, 11, 12, 13],  # left-arm
    ],
    "coco17": [
        [12, 14, 16],  # right-leg
        [11, 13, 15],  # left-leg
        [4, 2, 0, 1, 3],  # replace spine with head
        [6, 8, 10],  # right-arm
        [5, 7, 9],  # left-arm
    ],
}


def convert_motion_as_line_mesh(motion, skeleton_type="smpl22", const_color=None):
    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion)
    motion = motion.detach().cpu()
    kinematic_chain = KINEMATIC_CHAINS[skeleton_type]
    color_names = ["red", "green", "blue", "cyan", "magenta"]
    s_points = []
    e_points = []
    m_colors = []
    length = motion.shape[0]
    device = motion.device
    for chain, color_name in zip(kinematic_chain, color_names):
        num_line = len(chain) - 1
        s_points.append(motion[:, chain[:-1]])
        e_points.append(motion[:, chain[1:]])
        if const_color is not None:
            color_name = const_color
        color_ = get_const_colors(color_name, partial_shape=(length, num_line), alpha=1.0).to(device)  # (L, 4, 4)
        m_colors.append(color_[..., :3] * 255)  # (L, 4, 3)

    s_points = torch.cat(s_points, dim=1)  # (L, ?, 3)
    e_points = torch.cat(e_points, dim=1)
    m_colors = torch.cat(m_colors, dim=1)

    vertices = []
    for f in range(length):
        vertices_, faces, vertex_colors = create_skeleton_mesh(s_points[f], e_points[f], radius=0.02, color=m_colors[f])
        vertices.append(vertices_)
    vertices = torch.stack(vertices, dim=0)
    return vertices, faces, vertex_colors


def add_motion_as_lines(motion, wis3d, name="joints22", skeleton_type="smpl22", const_color=None, offset=0):
    """
    Args:
        motion (tensor): (L, J, 3)
    """
    vertices, faces, vertex_colors = convert_motion_as_line_mesh(
        motion, skeleton_type=skeleton_type, const_color=const_color
    )
    for f in range(len(vertices)):
        wis3d.set_scene_id(f + offset)
        wis3d.add_mesh(vertices[f], faces, vertex_colors, name=name)  # Add skeleton as cylinders
        # Old way to add lines, this may cause problems when the number of lines is large
        # wis3d.add_lines(s_points[f], e_points[f], m_colors[f], name=name)


def add_prog_motion_as_lines(motion, wis3d, name="joints22", skeleton_type="smpl22"):
    """
    Args:
        motion (tensor): (P, L, J, 3)
    """
    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion)
    P, L, J, _ = motion.shape
    device = motion.device

    kinematic_chain = KINEMATIC_CHAINS[skeleton_type]
    color_names = ["red", "green", "blue", "cyan", "magenta"]
    s_points = []
    e_points = []
    m_colors = []
    for chain, color_name in zip(kinematic_chain, color_names):
        num_line = len(chain) - 1
        s_points.append(motion[:, :, chain[:-1]])
        e_points.append(motion[:, :, chain[1:]])
        color_ = get_gradient_colors(color_name, L, alpha=1.0).to(device)  # (L, 4)
        color_ = color_[None, :, None, :].repeat(P, 1, num_line, 1)  # (P, L, num_line, 4)
        m_colors.append(color_[..., :3] * 255)  # (P, L, num_line, 3)
    s_points = torch.cat(s_points, dim=-2)  # (L, ?, 3)
    e_points = torch.cat(e_points, dim=-2)
    m_colors = torch.cat(m_colors, dim=-2)

    s_points = s_points.reshape(P, -1, 3)
    e_points = e_points.reshape(P, -1, 3)
    m_colors = m_colors.reshape(P, -1, 3)

    for p in range(P):
        wis3d.set_scene_id(p)
        wis3d.add_lines(s_points[p], e_points[p], m_colors[p], name=name)


def add_joints_motion_as_spheres(joints, wis3d, radius=0.05, name="joints", label_each_joint=False):
    """Visualize skeleton as spheres to explore the skeleton.
    Args:
        joints: (NF, NJ, 3)
        wis3d
        radius: radius of the spheres
        name
        label_each_joint: if True, each joints will have a label in wis3d (then you can interact with it, but it's slower)
    """
    colors = torch.zeros_like(joints).float()
    n_frames = joints.shape[0]
    n_joints = joints.shape[1]
    for i in range(n_joints):
        colors[:, i, 1] = 255 / n_joints * i
        colors[:, i, 2] = 255 / n_joints * (n_joints - i)
    for f in range(n_frames):
        wis3d.set_scene_id(f)
        if label_each_joint:
            for i in range(n_joints):
                wis3d.add_spheres(
                    joints[f, i].float(),
                    radius=radius,
                    colors=colors[f, i],
                    name=f"{name}-j{i}",
                )
        else:
            wis3d.add_spheres(
                joints[f].float(),
                radius=radius,
                colors=colors[f],
                name=f"{name}",
            )


def create_skeleton_mesh(p1, p2, radius, color, resolution=4, return_merged=True):
    """
    Create mesh between p1 and p2.
    Args:
        p1 (torch.Tensor): (N, 3),
        p2 (torch.Tensor): (N, 3),
        radius (float): radius,
        color (torch.Tensor): (N, 3)
        resolution (int): number of vertices in one circle, denoted as Q
    Returns:
        vertices (torch.Tensor): (N * 2Q, 3), if return_merged is False (N, 2Q, 3)
        faces (torch.Tensor): (M', 3), if return_merged is False (N, M, 3)
        vertex_colors (torch.Tensor): (N * 2Q, 3), if return_merged is False (N, 2Q, 3)
    """
    N = p1.shape[0]

    # Calculate segment direction
    seg_dir = p2 - p1  # (N, 3)
    unit_seg_dir = seg_dir / seg_dir.norm(dim=-1, keepdim=True)  # (N, 3)

    # Compute an orthogonal vector
    x_vec = torch.tensor([1, 0, 0], device=p1.device).float().unsqueeze(0).repeat(N, 1)  # (N, 3)
    y_vec = torch.tensor([0, 1, 0], device=p1.device).float().unsqueeze(0).repeat(N, 1)
    ortho_vec = torch.cross(unit_seg_dir, x_vec, dim=-1)  # (N, 3)
    ortho_vec_ = torch.cross(unit_seg_dir, y_vec, dim=-1)  # (N, 3)  backup
    ortho_vec = torch.where(ortho_vec.norm(dim=-1, keepdim=True) > 1e-3, ortho_vec, ortho_vec_)

    # Get circle points on two ends
    unit_ortho_vec = ortho_vec / ortho_vec.norm(dim=-1, keepdim=True)  # (N, 3)
    theta = torch.linspace(0, 2 * np.pi, resolution, device=p1.device)
    rotation_matrix = axis_angle_to_matrix(unit_seg_dir[:, None] * theta[None, :, None])  # (N, Q, 3, 3)
    rotated_points = einsum(rotation_matrix, unit_ortho_vec, "n q i j, n i -> n q j") * radius  # (N, Q, 3)
    bottom_points = rotated_points + p1.unsqueeze(1)  # (N, Q, 3)
    top_points = rotated_points + p2.unsqueeze(1)  # (N, Q, 3)

    # Combine bottom and top points
    vertices = torch.cat([bottom_points, top_points], dim=1)  # (N, 2Q, 3)

    # Generate face
    indices = torch.arange(0, resolution, device=p1.device)
    bottom_indices = indices
    top_indices = indices + resolution

    # outside face
    face_bottom = torch.stack([bottom_indices[:-2], bottom_indices[1:-1], bottom_indices[-1].repeat(resolution - 2)], 1)
    face_top = torch.stack([top_indices[1:-1], top_indices[:-2], top_indices[-1].repeat(resolution - 2)], 1)
    faces = torch.cat(
        [
            torch.stack([bottom_indices[1:], bottom_indices[:-1], top_indices[:-1]], 1),  # out face
            torch.stack([bottom_indices[1:], top_indices[:-1], top_indices[1:]], 1),  # out face
            face_bottom,
            face_top,
        ]
    )
    faces = faces.unsqueeze(0).repeat(p1.shape[0], 1, 1)  # (N, M, 3)

    # Assign colors
    vertex_colors = color.unsqueeze(1).repeat(1, resolution * 2, 1)

    if return_merged:
        # manully adjust face ids
        N, V = vertices.shape[:2]
        faces = faces + torch.arange(0, N, device=p1.device).unsqueeze(1).unsqueeze(1) * V
        faces = faces.reshape(-1, 3)
        vertices = vertices.reshape(-1, 3)
        vertex_colors = vertex_colors.reshape(-1, 3)

    return vertices, faces, vertex_colors


def get_lines_of_my_frustum(frustum_points):
    """
    frustum_points: (B, 8, 3), in (near {lu ru rd ld}, far {lu ru rd ld})
    """
    start_points = frustum_points[:, [0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7]].cpu().numpy()
    end_points = frustum_points[:, [4, 5, 6, 7, 1, 2, 3, 0, 5, 6, 7, 4]].cpu().numpy()
    return start_points, end_points


def draw_colored_vec(wis3d, vec, name, radius=0.02, colors="r", starts=None, l=1.0):
    """
    Args:
        vec: (3) or (L, 3), should be the same length as colors, like 'rgb'
    """
    if len(vec.shape) == 1:
        vec = vec[None]
    else:
        assert len(vec.shape) == 2

    assert len(vec) == len(colors)
    # split colors, 'rgb' to 'r', 'g', 'b'
    color_tensor = torch.zeros((len(colors), 3))
    c2rgb = {
        "r": torch.tensor([1, 0, 0]).float(),
        "g": torch.tensor([0, 1, 0]).float(),
        "b": torch.tensor([0, 0, 1]).float(),
    }
    for i, c in enumerate(colors):
        color_tensor[i] = c2rgb[c]

    if starts is None:
        starts = torch.zeros_like(vec)
    ends = starts + vec * l
    vertices, faces, vertex_colors = create_skeleton_mesh(starts, ends, radius, color_tensor, resolution=10)
    wis3d.add_mesh(vertices, faces, vertex_colors, name=name)


def draw_T_w2c(wis3d, T_w2c, name, radius=0.01, all_in_one=True, l=0.1):
    """
    Draw a camera trajectory in world coordinate.
    Args:
        T_w2c: (L, 4, 4)
    """
    color_tensor = torch.eye(3)
    if all_in_one:
        starts = -T_w2c[:, :3, :3].mT @ T_w2c[:, :3, [3]]  # (L, 3, 1)
        starts = starts[:, None, :, 0].expand(-1, 3, -1).reshape(-1, 3)  # (L*3, 3)
        vec = T_w2c[:, :3, :3].reshape(-1, 3)  # (L * 3, 3)
        ends = starts + vec * l
        color_tensor = color_tensor[None].expand(T_w2c.size(0), -1, -1).reshape(-1, 3)

        vertices, faces, vertex_colors = create_skeleton_mesh(starts, ends, radius, color_tensor, resolution=10)
    else:
        raise NotImplementedError
    wis3d.add_mesh(vertices, faces, vertex_colors, name=name)


def create_checkerboard_mesh(y=0.0, grid_size=1.0, bounds=((-3, -3), (3, 3))):
    """
    example usage:
        vertices, faces, vertex_colors = create_checkerboard_mesh()
        wis3d.add_mesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors, name="one")
    """
    color1 = np.array([236, 240, 241], np.uint8)  # light
    color2 = np.array([120, 120, 120], np.uint8)  # dark

    # 扩大范围
    min_x, min_z = bounds[0]
    max_x, max_z = bounds[1]
    min_x = grid_size * np.floor(min_x / grid_size)
    min_z = grid_size * np.floor(min_z / grid_size)
    max_x = grid_size * np.ceil(max_x / grid_size)
    max_z = grid_size * np.ceil(max_z / grid_size)

    vertices = []
    faces = []
    vertex_colors = []
    eps = 1e-4  # HACK: disable smooth color & double-side color artifacts of wis3d

    for i, x in enumerate(np.arange(min_x, max_x, grid_size)):
        for j, z in enumerate(np.arange(min_z, max_z, grid_size)):

            # Right-hand rule for normal direction
            x += ((i % 2 * 2) - 1) * eps
            z += ((j % 2 * 2) - 1) * eps
            v1 = np.array([x, y, z])
            v2 = np.array([x, y, z + grid_size])
            v3 = np.array([x + grid_size, y, z + grid_size])
            v4 = np.array([x + grid_size, y, z])
            offset = np.array([0, -eps, 0])  # For visualizing the down-side of the mesh

            vertices.extend([v1, v2, v3, v4, v1 + offset, v2 + offset, v3 + offset, v4 + offset])
            idx = len(vertices) - 8
            faces.extend(
                [
                    [idx, idx + 1, idx + 2],
                    [idx + 2, idx + 3, idx],
                    [idx + 4, idx + 7, idx + 6],  # double-sided
                    [idx + 6, idx + 5, idx + 4],  # double-sided
                ]
            )
            vertex_color = color1 if (i + j) % 2 == 0 else color2
            vertex_colors.extend([vertex_color] * 8)

    # To numpy.array and the shape should be (n, 3)
    vertices = np.array(vertices)
    faces = np.array(faces)
    vertex_colors = np.array(vertex_colors)
    assert len(vertices.shape) == 2 and vertices.shape[1] == 3
    assert len(faces.shape) == 2 and faces.shape[1] == 3
    assert len(vertex_colors.shape) == 2 and vertex_colors.shape[1] == 3 and vertex_colors.dtype == np.uint8

    return vertices, faces, vertex_colors


def add_a_trimesh(mesh, wis3d, name):
    mesh.apply_transform(wis3d.three_to_world)

    # filename = wis3d.__get_export_file_name("mesh", name)
    export_dir = Path(wis3d.out_folder) / wis3d.sequence_name / f"{wis3d.scene_id:05d}" / "meshes"
    export_dir.mkdir(parents=True, exist_ok=True)
    assert name is not None
    filename = export_dir / f"{name}.ply"
    wis3d.counters["mesh"] += 1

    mesh.export(filename)
