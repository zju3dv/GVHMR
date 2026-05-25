import os
import random
import numpy as np
import torch
import smplx
import pyrender
import trimesh
import cv2
from tqdm import tqdm

# ── 설정 ────────────────────────────────────────
SMPLX_MODEL_PATH = r'C:\Users\user\Desktop\CG\models'
AMASS_ROOT       = r'C:\Users\user\Desktop\CG\캡스톤\CMU'
OUTPUT_ROOT      = r'C:\Users\user\Desktop\CG\캡스톤\output'

NUM_SAMPLES      = 20
TARGET_FPS       = 30
IMG_W, IMG_H     = 3840, 2160
SEED             = 42

# 카메라 관련
CAM_DIST                 = 4.2
MIN_CAM_DIST             = 3.8
CAM_HEIGHT_RATIO         = 0.10   # 카메라를 중심보다 약간 위로
TARGET_HEIGHT_RATIO      = 0.05   # 인체 중심보다 약간 위를 바라보게
FOV_Y                    = np.pi / 4.3

# 좌표계 변환
# 현재 스크린샷 기준으로는 Z-up 데이터를 Y-up으로 바꾸는 게 맞을 가능성이 큼
CONVERT_Z_UP_TO_Y_UP     = True

# 밝기 관련
AMBIENT_INTENSITY        = 0.24
MAIN_LIGHT_INTENSITY     = 2.2
FILL_LIGHT_INTENSITY     = 0.75
EXPOSURE_GAIN            = 0.92

# 배경색
BG_COLOR = [0.78, 0.78, 0.78, 1.0]

# 기존 영상 덮어쓰기
OVERWRITE_EXISTING = True
# ────────────────────────────────────────────────

os.makedirs(os.path.join(OUTPUT_ROOT, 'videos'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, 'gt'),     exist_ok=True)


# ── 유틸: 벡터 정규화 ────────────────────────────
def normalize(v, eps=1e-8):
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n


# ── look-at 카메라 pose 생성 ─────────────────────
def lookat_pose(cam_pos, target, up=np.array([0.0, 1.0, 0.0], dtype=np.float64)):
    """
    pyrender용 카메라 pose 생성.
    pyrender에서 카메라의 -Z 축이 전방이다.
    """
    cam_pos = np.asarray(cam_pos, dtype=np.float64)
    target  = np.asarray(target,  dtype=np.float64)
    up      = np.asarray(up,      dtype=np.float64)

    forward = normalize(target - cam_pos)

    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-8:
        # forward와 up이 거의 평행할 때 예외 처리
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, up)

    right   = normalize(right)
    true_up = normalize(np.cross(right, forward))

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = cam_pos
    return pose


# ── 좌표축 변환: Z-up → Y-up ─────────────────────
def convert_vertices_to_y_up(vertices):
    """
    vertices: (N, V, 3)
    Z-up 좌표계를 Y-up으로 바꿈.
    (x, y, z) -> (x, z, -y)
    즉 X축 기준 -90도 회전과 동일.
    """
    if not CONVERT_Z_UP_TO_Y_UP:
        return vertices

    R = np.array([
        [1.0, 0.0,  0.0],
        [0.0, 0.0,  1.0],
        [0.0,-1.0,  0.0]
    ], dtype=np.float32)

    # (N, V, 3) @ (3, 3)^T
    return vertices @ R.T


# ── bbox 기반 중심 / 높이 계산 ───────────────────
def get_sequence_center_height(vertices):
    """
    vertices: (N, V, 3)
    전체 시퀀스 기준 bbox 중심과 높이 계산
    """
    all_verts = vertices.reshape(-1, 3)

    vmin = np.min(all_verts, axis=0)
    vmax = np.max(all_verts, axis=0)

    center = (vmin + vmax) * 0.5
    height = float(vmax[1] - vmin[1])

    if height < 1e-6:
        height = 1.7

    return center, height, vmin, vmax


# ── 동서남북 카메라 pose 생성 ────────────────────
def get_camera_poses(center, height, dist=CAM_DIST):
    """
    메시가 Y-up으로 정렬된 상태라고 가정.
    카메라는 Y축 기준으로 수평 회전하며 동서남북 4방향에서 촬영.
    """
    cx, cy, cz = center

    cam_dist = max(dist, MIN_CAM_DIST, height * 2.0)
    cam_y    = cy + height * CAM_HEIGHT_RATIO

    target = np.array([
        cx,
        cy + height * TARGET_HEIGHT_RATIO,
        cz
    ], dtype=np.float64)

    # 진짜 동서남북: Y축 기준으로 X/Z 평면 위에서만 이동
    cam_positions = {
        'north': np.array([cx,            cam_y, cz + cam_dist], dtype=np.float64),
        'east' : np.array([cx + cam_dist, cam_y, cz           ], dtype=np.float64),
        'south': np.array([cx,            cam_y, cz - cam_dist], dtype=np.float64),
        'west' : np.array([cx - cam_dist, cam_y, cz           ], dtype=np.float64),
    }

    cam_poses = {
        name: lookat_pose(pos, target, up=np.array([0.0, 1.0, 0.0]))
        for name, pos in cam_positions.items()
    }

    # 보조광: 살짝 위 + 대각선에서 비춤
    fill_pos = np.array([
        cx - cam_dist * 0.6,
        cy + height   * 1.2,
        cz + cam_dist * 0.6
    ], dtype=np.float64)

    fill_light_pose = lookat_pose(fill_pos, target, up=np.array([0.0, 1.0, 0.0]))

    return cam_poses, fill_light_pose, target, cam_dist


# ── AMASS 로드 ───────────────────────────────────
def load_amass(filepath, target_fps=30):
    try:
        data = np.load(filepath, allow_pickle=True)

        if 'mocap_framerate' in data:
            orig_fps = int(data['mocap_framerate'])
        elif 'mocap_frame_rate' in data:
            orig_fps = int(data['mocap_frame_rate'])
        else:
            orig_fps = 120

        step  = max(1, orig_fps // target_fps)
        poses = data['poses'][::step]
        trans = data['trans'][::step]
        betas = data['betas'][:10]

        gender = data['gender']
        if isinstance(gender, np.ndarray):
            gender = gender.item()
        if isinstance(gender, bytes):
            gender = gender.decode('utf-8')
        gender = str(gender).lower()

        if gender not in ['male', 'female']:
            gender = 'neutral'

        if len(poses) < 30:
            return None

        return poses, trans, betas, gender

    except Exception as e:
        print(f'로드 실패: {filepath} - {e}')
        return None


# ── SMPL-X 메시 생성 ─────────────────────────────
def get_vertices(poses, trans, betas, gender):
    N = len(poses)

    model = smplx.create(
        SMPLX_MODEL_PATH,
        model_type='smplx',
        gender=gender,
        use_pca=False,
        batch_size=N
    )

    with torch.no_grad():
        output = model(
            betas=torch.tensor(betas, dtype=torch.float32).unsqueeze(0).expand(N, -1),
            global_orient=torch.tensor(poses[:, :3],        dtype=torch.float32),
            body_pose=torch.tensor(poses[:, 3:66],           dtype=torch.float32),
            left_hand_pose=torch.tensor(poses[:, 75:120],    dtype=torch.float32),
            right_hand_pose=torch.tensor(poses[:, 120:165],  dtype=torch.float32),
            transl=torch.tensor(trans, dtype=torch.float32),
            return_verts=True
        )

    vertices = output.vertices.cpu().numpy()
    faces    = model.faces
    return vertices, faces


# ── 단일 시점 렌더링 ─────────────────────────────
def render_one_view(vertices, faces, cam_pose, fill_light_pose):
    renderer = pyrender.OffscreenRenderer(IMG_W, IMG_H)

    camera = pyrender.PerspectiveCamera(
        yfov=FOV_Y,
        aspectRatio=IMG_W / IMG_H
    )

    frames = []

    try:
        for verts in vertices:
            scene = pyrender.Scene(
                ambient_light=[
                    AMBIENT_INTENSITY,
                    AMBIENT_INTENSITY,
                    AMBIENT_INTENSITY
                ],
                bg_color=BG_COLOR
            )

            mesh = trimesh.Trimesh(
                vertices=verts,
                faces=faces,
                process=False
            )

            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.72, 0.56, 0.49, 1.0],
                metallicFactor=0.0,
                roughnessFactor=0.72
            )

            render_mesh = pyrender.Mesh.from_trimesh(
                mesh,
                material=material,
                smooth=True
            )

            scene.add(render_mesh)
            scene.add(camera, pose=cam_pose)

            # 주광: 카메라 방향과 같은 방향에서 부드럽게 비춤
            main_light = pyrender.DirectionalLight(
                color=np.ones(3),
                intensity=MAIN_LIGHT_INTENSITY
            )
            scene.add(main_light, pose=cam_pose)

            # 보조광: 위/대각선에서 약하게 비춤
            fill_light = pyrender.DirectionalLight(
                color=np.ones(3),
                intensity=FILL_LIGHT_INTENSITY
            )
            scene.add(fill_light, pose=fill_light_pose)

            color, _ = renderer.render(
                scene,
                flags=pyrender.RenderFlags.RGBA
            )

            rgb = color[:, :, :3].astype(np.float32)
            rgb = np.clip(rgb * EXPOSURE_GAIN, 0, 255).astype(np.uint8)

            frames.append(rgb)

    finally:
        renderer.delete()

    return frames


# ── 영상 저장 ────────────────────────────────────
def save_video(frames, path, fps):
    if len(frames) == 0:
        print(f'저장 실패: 프레임 없음 - {path}')
        return

    h, w = frames[0].shape[:2]

    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h)
    )

    if not writer.isOpened():
        print(f'VideoWriter 열기 실패: {path}')
        return

    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    writer.release()


# ── GT 저장 ──────────────────────────────────────
def save_gt(poses, trans, betas, gender, path):
    np.savez(
        path,
        global_orient   = poses[:, :3],
        body_pose       = poses[:, 3:66],
        left_hand_pose  = poses[:, 75:120],
        right_hand_pose = poses[:, 120:165],
        betas           = betas,
        transl          = trans,
        gender          = gender
    )


# ── npz 파일 수집 ────────────────────────────────
def collect_npz_files(root):
    files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith('stageii.npz'):
                files.append(os.path.join(dirpath, f))
    return files


# ── 메인 ─────────────────────────────────────────
def main():
    random.seed(SEED)
    all_files = collect_npz_files(AMASS_ROOT)

    if not all_files:
        print(f"'{AMASS_ROOT}' 경로에서 npz 파일을 찾을 수 없습니다.")
        return

    sampled_files = random.sample(all_files, min(NUM_SAMPLES, len(all_files)))
    print(f'총 {len(all_files)}개 중 {len(sampled_files)}개 선택\n')

    success = 0

    for filepath in tqdm(sampled_files, desc='시퀀스 전체 진행'):
        result = load_amass(filepath, TARGET_FPS)
        if result is None:
            continue

        poses, trans, betas, gender = result
        rel_path = os.path.relpath(filepath, AMASS_ROOT)
        name     = rel_path.replace('/', '_').replace('\\', '_').replace('.npz', '')

        try:
            vertices, faces = get_vertices(poses, trans, betas, gender)

            # 좌표축 보정: Z-up -> Y-up
            vertices = convert_vertices_to_y_up(vertices)

            # bbox 기반 중심/높이 계산
            center, height, vmin, vmax = get_sequence_center_height(vertices)

            # 카메라 4방향 생성
            cam_poses, fill_light_pose, target, cam_dist = get_camera_poses(
                center=center,
                height=height,
                dist=CAM_DIST
            )

            print(f'\n[{name}]')
            print(f'  bbox min      : {vmin}')
            print(f'  bbox max      : {vmax}')
            print(f'  center        : {center}')
            print(f'  height(y)     : {height:.4f}')
            print(f'  camera target : {target}')
            print(f'  camera dist   : {cam_dist:.4f}')

            for view_name, cam_pose in cam_poses.items():
                video_path = os.path.join(OUTPUT_ROOT, 'videos', f'{name}_{view_name}.mp4')

                if os.path.exists(video_path) and not OVERWRITE_EXISTING:
                    print(f'  [{view_name}] 이미 존재, 스킵')
                    continue

                frames = render_one_view(
                    vertices=vertices,
                    faces=faces,
                    cam_pose=cam_pose,
                    fill_light_pose=fill_light_pose
                )

                save_video(frames, video_path, TARGET_FPS)
                print(f'  [{view_name}] 저장 완료: {video_path}')

            gt_path = os.path.join(OUTPUT_ROOT, 'gt', f'{name}.npz')
            if not os.path.exists(gt_path) or OVERWRITE_EXISTING:
                save_gt(poses, trans, betas, gender, gt_path)

            success += 1
            print(f'[{success}/{len(sampled_files)}] 완료: {name}\n')

        except Exception as e:
            import traceback
            print(f'실패: {name} - {e}')
            traceback.print_exc()

    print(f'완료: {success}/{len(sampled_files)}개')
    print(f'영상 총 {success * 4}개 생성됨')
    print(f'결과 위치: {OUTPUT_ROOT}')


if __name__ == "__main__":
    main()