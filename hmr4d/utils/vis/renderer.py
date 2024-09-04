import cv2
import torch
import numpy as np

from pytorch3d.renderer import (
    PerspectiveCameras,
    TexturesVertex,
    PointLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
)
from pytorch3d.structures import Meshes
from pytorch3d.structures.meshes import join_meshes_as_scene
from pytorch3d.renderer.cameras import look_at_rotation
from pytorch3d.transforms import axis_angle_to_matrix

from .renderer_tools import get_colors, checkerboard_geometry


colors_str_map = {
    "gray": [0.8, 0.8, 0.8],
    "green": [39, 194, 128],
}


def overlay_image_onto_background(image, mask, bbox, background):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    out_image = background.copy()
    bbox = bbox[0].int().cpu().numpy().copy()
    roi_image = out_image[bbox[1] : bbox[3], bbox[0] : bbox[2]]

    roi_image[mask] = image[mask]
    out_image[bbox[1] : bbox[3], bbox[0] : bbox[2]] = roi_image

    return out_image


def update_intrinsics_from_bbox(K_org, bbox):
    device, dtype = K_org.device, K_org.dtype

    K = torch.zeros((K_org.shape[0], 4, 4)).to(device=device, dtype=dtype)
    K[:, :3, :3] = K_org.clone()
    K[:, 2, 2] = 0
    K[:, 2, -1] = 1
    K[:, -1, 2] = 1

    image_sizes = []
    for idx, bbox in enumerate(bbox):
        left, upper, right, lower = bbox
        cx, cy = K[idx, 0, 2], K[idx, 1, 2]

        new_cx = cx - left
        new_cy = cy - upper
        new_height = max(lower - upper, 1)
        new_width = max(right - left, 1)
        new_cx = new_width - new_cx
        new_cy = new_height - new_cy

        K[idx, 0, 2] = new_cx
        K[idx, 1, 2] = new_cy
        image_sizes.append((int(new_height), int(new_width)))

    return K, image_sizes


def perspective_projection(x3d, K, R=None, T=None):
    if R != None:
        x3d = torch.matmul(R, x3d.transpose(1, 2)).transpose(1, 2)
    if T != None:
        x3d = x3d + T.transpose(1, 2)

    x2d = torch.div(x3d, x3d[..., 2:])
    x2d = torch.matmul(K, x2d.transpose(-1, -2)).transpose(-1, -2)[..., :2]
    return x2d


def compute_bbox_from_points(X, img_w, img_h, scaleFactor=1.2):
    left = torch.clamp(X.min(1)[0][:, 0], min=0, max=img_w)
    right = torch.clamp(X.max(1)[0][:, 0], min=0, max=img_w)
    top = torch.clamp(X.min(1)[0][:, 1], min=0, max=img_h)
    bottom = torch.clamp(X.max(1)[0][:, 1], min=0, max=img_h)

    cx = (left + right) / 2
    cy = (top + bottom) / 2
    width = right - left
    height = bottom - top

    new_left = torch.clamp(cx - width / 2 * scaleFactor, min=0, max=img_w - 1)
    new_right = torch.clamp(cx + width / 2 * scaleFactor, min=1, max=img_w)
    new_top = torch.clamp(cy - height / 2 * scaleFactor, min=0, max=img_h - 1)
    new_bottom = torch.clamp(cy + height / 2 * scaleFactor, min=1, max=img_h)

    bbox = torch.stack((new_left.detach(), new_top.detach(), new_right.detach(), new_bottom.detach())).int().float().T

    return bbox


class Renderer:
    def __init__(self, width, height, focal_length=None, device="cuda", faces=None, K=None, bin_size=None):
        """set bin_size to 0 for no binning"""
        self.width = width
        self.height = height
        self.bin_size = bin_size
        assert (focal_length is not None) ^ (K is not None), "focal_length and K are mutually exclusive"

        self.device = device
        if faces is not None:
            if isinstance(faces, np.ndarray):
                faces = torch.from_numpy((faces).astype("int"))
            self.faces = faces.unsqueeze(0).to(self.device)

        self.initialize_camera_params(focal_length, K)
        self.lights = PointLights(device=device, location=[[0.0, 0.0, -10.0]])
        self.create_renderer()

    def create_renderer(self):
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=RasterizationSettings(
                    image_size=self.image_sizes[0], blur_radius=1e-5, bin_size=self.bin_size
                ),
            ),
            shader=SoftPhongShader(
                device=self.device,
                lights=self.lights,
            ),
        )

    def create_camera(self, R=None, T=None):
        if R is not None:
            self.R = R.clone().view(1, 3, 3).to(self.device)
        if T is not None:
            self.T = T.clone().view(1, 3).to(self.device)

        return PerspectiveCameras(
            device=self.device, R=self.R.mT, T=self.T, K=self.K_full, image_size=self.image_sizes, in_ndc=False
        )

    def initialize_camera_params(self, focal_length, K):
        # Extrinsics
        self.R = torch.diag(torch.tensor([1, 1, 1])).float().to(self.device).unsqueeze(0)

        self.T = torch.tensor([0, 0, 0]).unsqueeze(0).float().to(self.device)

        # Intrinsics
        if K is not None:
            self.K = K.float().reshape(1, 3, 3).to(self.device)
        else:
            assert focal_length is not None, "focal_length or K should be provided"
            self.K = (
                torch.tensor([[focal_length, 0, self.width / 2], [0, focal_length, self.height / 2], [0, 0, 1]])
                .float()
                .reshape(1, 3, 3)
                .to(self.device)
            )
        self.bboxes = torch.tensor([[0, 0, self.width, self.height]]).float()
        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, self.bboxes)
        self.cameras = self.create_camera()

    def set_intrinsic(self, K):
        self.K = K.reshape(1, 3, 3)

    def set_ground(self, length, center_x, center_z):
        device = self.device
        length, center_x, center_z = map(float, (length, center_x, center_z))
        v, f, vc, fc = map(torch.from_numpy, checkerboard_geometry(length=length, c1=center_x, c2=center_z, up="y"))
        v, f, vc = v.to(device), f.to(device), vc.to(device)
        self.ground_geometry = [v, f, vc]

    def update_bbox(self, x3d, scale=2.0, mask=None):
        """Update bbox of cameras from the given 3d points

        x3d: input 3D keypoints (or vertices), (num_frames, num_points, 3)
        """

        if x3d.size(-1) != 3:
            x2d = x3d.unsqueeze(0)
        else:
            x2d = perspective_projection(x3d.unsqueeze(0), self.K, self.R, self.T.reshape(1, 3, 1))

        if mask is not None:
            x2d = x2d[:, ~mask]

        bbox = compute_bbox_from_points(x2d, self.width, self.height, scale)
        self.bboxes = bbox

        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, bbox)
        self.cameras = self.create_camera()
        self.create_renderer()

    def reset_bbox(
        self,
    ):
        bbox = torch.zeros((1, 4)).float().to(self.device)
        bbox[0, 2] = self.width
        bbox[0, 3] = self.height
        self.bboxes = bbox

        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, bbox)
        self.cameras = self.create_camera()
        self.create_renderer()

    def render_mesh(self, vertices, background=None, colors=[0.8, 0.8, 0.8], VI=50):
        self.update_bbox(vertices[::VI], scale=1.2)
        vertices = vertices.unsqueeze(0)

        if isinstance(colors, torch.Tensor):
            # per-vertex color
            verts_features = colors.to(device=vertices.device, dtype=vertices.dtype)
            colors = [0.8, 0.8, 0.8]
        else:
            if colors[0] > 1:
                colors = [c / 255.0 for c in colors]
            verts_features = torch.tensor(colors).reshape(1, 1, 3).to(device=vertices.device, dtype=vertices.dtype)
            verts_features = verts_features.repeat(1, vertices.shape[1], 1)
        textures = TexturesVertex(verts_features=verts_features)

        mesh = Meshes(
            verts=vertices,
            faces=self.faces,
            textures=textures,
        )

        materials = Materials(device=self.device, specular_color=(colors,), shininess=0)

        results = torch.flip(self.renderer(mesh, materials=materials, cameras=self.cameras, lights=self.lights), [1, 2])
        image = results[0, ..., :3] * 255
        mask = results[0, ..., -1] > 1e-3

        if background is None:
            background = np.ones((self.height, self.width, 3)).astype(np.uint8) * 255

        image = overlay_image_onto_background(image, mask, self.bboxes, background.copy())
        self.reset_bbox()
        return image

    def render_with_ground(self, verts, colors, cameras, lights, faces=None):
        """
        :param verts (N, V, 3), potential multiple people
        :param colors (N, 3) or (N, V, 3)
        :param faces (N, F, 3), optional, otherwise self.faces is used will be used
        """
        # Sanity check of input verts, colors and faces: (B, V, 3), (B, F, 3), (B, V, 3)
        N, V, _ = verts.shape
        if faces is None:
            faces = self.faces.clone().expand(N, -1, -1)
        else:
            assert len(faces.shape) == 3, "faces should have shape of (N, F, 3)"

        assert len(colors.shape) in [2, 3]
        if len(colors.shape) == 2:
            assert len(colors) == N, "colors of shape 2 should be (N, 3)"
            colors = colors[:, None]
        colors = colors.expand(N, V, -1)[..., :3]

        # (V, 3), (F, 3), (V, 3)
        gv, gf, gc = self.ground_geometry
        verts = list(torch.unbind(verts, dim=0)) + [gv]
        faces = list(torch.unbind(faces, dim=0)) + [gf]
        colors = list(torch.unbind(colors, dim=0)) + [gc[..., :3]]
        mesh = create_meshes(verts, faces, colors)

        materials = Materials(device=self.device, shininess=0)

        results = self.renderer(mesh, cameras=cameras, lights=lights, materials=materials)
        image = (results[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)

        return image


def create_meshes(verts, faces, colors):
    """
    :param verts (B, V, 3)
    :param faces (B, F, 3)
    :param colors (B, V, 3)
    """
    textures = TexturesVertex(verts_features=colors)
    meshes = Meshes(verts=verts, faces=faces, textures=textures)
    return join_meshes_as_scene(meshes)


def get_global_cameras(verts, device="cuda", distance=5, position=(-5.0, 5.0, 0.0)):
    """This always put object at the center of view"""
    positions = torch.tensor([position]).repeat(len(verts), 1)
    targets = verts.mean(1)

    directions = targets - positions
    directions = directions / torch.norm(directions, dim=-1).unsqueeze(-1) * distance
    positions = targets - directions

    rotation = look_at_rotation(positions, targets).mT
    translation = -(rotation @ positions.unsqueeze(-1)).squeeze(-1)

    lights = PointLights(device=device, location=[position])
    return rotation, translation, lights


def get_global_cameras_static(
    verts, beta=4.0, cam_height_degree=30, target_center_height=1.0, use_long_axis=False, vec_rot=45, device="cuda"
):
    L, V, _ = verts.shape

    # Compute target trajectory, denote as center + scale
    targets = verts.mean(1)  # (L, 3)
    targets[:, 1] = 0  # project to xz-plane
    target_center = targets.mean(0)  # (3,)
    target_scale, target_idx = torch.norm(targets - target_center, dim=-1).max(0)

    # a 45 degree vec from longest axis
    if use_long_axis:
        long_vec = targets[target_idx] - target_center  # (x, 0, z)
        long_vec = long_vec / torch.norm(long_vec)
        R = axis_angle_to_matrix(torch.tensor([0, np.pi / 4, 0])).to(long_vec)
        vec = R @ long_vec
    else:
        vec_rad = vec_rot / 180 * np.pi
        vec = torch.tensor([np.sin(vec_rad), 0, np.cos(vec_rad)]).float()
        vec = vec / torch.norm(vec)

    # Compute camera position (center + scale * vec * beta) + y=4
    target_scale = max(target_scale, 1.0) * beta
    position = target_center + vec * target_scale
    position[1] = target_scale * np.tan(np.pi * cam_height_degree / 180) + target_center_height

    # Compute camera rotation and translation
    positions = position.unsqueeze(0).repeat(L, 1)
    target_centers = target_center.unsqueeze(0).repeat(L, 1)
    target_centers[:, 1] = target_center_height
    rotation = look_at_rotation(positions, target_centers).mT
    translation = -(rotation @ positions.unsqueeze(-1)).squeeze(-1)

    lights = PointLights(device=device, location=[position.tolist()])
    return rotation, translation, lights


def get_ground_params_from_points(root_points, vert_points):
    """xz-plane is the ground plane
    Args:
        root_points: (L, 3), to decide center
        vert_points: (L, V, 3), to decide scale
    """
    root_max = root_points.max(0)[0]  # (3,)
    root_min = root_points.min(0)[0]  # (3,)
    cx, _, cz = (root_max + root_min) / 2.0

    vert_max = vert_points.reshape(-1, 3).max(0)[0]  # (L, 3)
    vert_min = vert_points.reshape(-1, 3).min(0)[0]  # (L, 3)
    scale = (vert_max - vert_min)[[0, 2]].max()
    return float(scale), float(cx), float(cz)
