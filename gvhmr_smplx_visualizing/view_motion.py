# -*- coding: utf-8 -*-
from __future__ import print_function

from hand_temporal_model import TemporalSmoother
from hand_kalman_filter import HandPoseKalmanFilter

import time
import numpy as np
import torch
import open3d as o3d
import smplx
import os
import cv2
import pickle

# --- 경로 및 설정 ---
BASE_OUTPUT_DIR = r""
HAMER_PATH      = os.path.join(BASE_OUTPUT_DIR, "hamer_out")
PT_PATH         = os.path.join(BASE_OUTPUT_DIR, "hmr4d_results.pt")
MODEL_PATH      = "models"
PARAM_KEY       = "smpl_params_global"
GENDER          = "neutral"
FPS             = 30

TEMPORAL_MODEL_PATH = "hand_temporal_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────────────────

def load_motion(pt_path, param_key="smpl_params_global"):
    data   = torch.load(pt_path, map_location=device)
    params = data[param_key]
    return {
        "body_pose":     params["body_pose"].float(),
        "betas":         params["betas"].float(),
        "global_orient": params["global_orient"].float(),
        "transl":        params["transl"].float(),
    }


def build_smpl_model(model_path, gender="neutral"):
    return smplx.create(
        model_path=model_path,
        model_type="smplh",
        gender=gender,
        use_pca=False,
        num_left_hand_pca_comps=45,
        num_right_hand_pca_comps=45,
        flat_hand_mean=True,
        batch_size=1,
        ext="pkl",
    ).to(device)


def load_hamer_hands(frame_idx):
    prefix     = "{:04d}_".format(frame_idx)
    hand_files = [
        f for f in os.listdir(HAMER_PATH)
        if f.startswith(prefix) and f.endswith(".npz")
    ]

    l_hand, r_hand = None, None
    l_conf, r_conf = 0.0, 0.0

    for f in hand_files:
        try:
            data            = np.load(os.path.join(HAMER_PATH, f), allow_pickle=True)
            is_right        = float(data["is_right"])
            hand_pose_mat   = data["mano_params"].item()["hand_pose"]
            conf            = float(data["keypoint_confidiences"]) if "keypoint_confidiences" in data else 1.0

            aa_list = []
            for j in range(15):
                aa, _ = cv2.Rodrigues(hand_pose_mat[j])
                aa    = aa.flatten()
                if is_right == 0:           # 왼손 미러 보정
                    aa[1] = -aa[1]
                    aa[2] = -aa[2]
                aa_list.append(aa)

            pose = torch.tensor(np.array(aa_list)).float().reshape(1, 45).to(device)

            if is_right == 1:
                r_hand, r_conf = pose, conf
            else:
                l_hand, l_conf = pose, conf

        except Exception as e:
            print(f"[WARN] {f} 로드 실패: {e}")

    return l_hand, r_hand, l_conf, r_conf


# ──────────────────────────────────────────────────────────
# SMPL-H 메쉬
# ──────────────────────────────────────────────────────────

def make_mesh_from_frame(model, body_pose, betas, global_orient, transl,
                         left_hand_pose=None, right_hand_pose=None):
    with torch.no_grad():
        lh = (left_hand_pose  if left_hand_pose  is not None else torch.zeros(1, 45)).to(device)
        rh = (right_hand_pose if right_hand_pose is not None else torch.zeros(1, 45)).to(device)

        output = model(
            betas=betas.reshape(1, -1),
            global_orient=global_orient.reshape(1, -1),
            body_pose=body_pose.reshape(1, -1),
            transl=transl.reshape(1, -1),
            left_hand_pose=lh,
            right_hand_pose=rh,
            return_verts=True,
        )

    vertices      = output.vertices[0].cpu().numpy()
    faces         = model.faces.astype(np.int32)
    right_wrist   = output.joints[0, 21].cpu().numpy()
    return vertices, faces, right_wrist


def create_open3d_mesh(vertices, faces):
    mesh           = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    return mesh


def create_studio_room():
    floor = o3d.geometry.TriangleMesh.create_box(width=10, height=0.01, depth=10)
    floor.translate([-5, -1.0, -5])
    floor.paint_uniform_color([0.3, 0.3, 0.3])

    back_wall = o3d.geometry.TriangleMesh.create_box(width=10, height=5, depth=0.01)
    back_wall.translate([-5, -1.0, -5])
    back_wall.paint_uniform_color([0.2, 0.2, 0.2])

    return [floor, back_wall]


# ──────────────────────────────────────────────────────────
# 전처리: raw → Kalman → Temporal
# ──────────────────────────────────────────────────────────

def preprocess_hands(T):
    """모든 프레임의 손 데이터를 미리 로드하고 필터링합니다."""

    # 1) Raw 로드
    print("[INFO] 전체 프레임 손 데이터 로드 중...")
    all_l_raw,  all_r_raw  = [], []
    all_l_conf, all_r_conf = [], []

    for i in range(T):
        l, r, lc, rc = load_hamer_hands(i)
        all_l_raw.append(l)
        all_r_raw.append(r)
        all_l_conf.append(lc)
        all_r_conf.append(rc)

    # 2) Kalman Filter
    print("[INFO] Kalman Filter 처리 중...")
    lh_kalman = HandPoseKalmanFilter(
        fps=FPS, process_var=1e-4, obs_var=1e-2,
        confidence_power=2.0, outlier_threshold=0.8,
        outlier_noise_scale=10.0, device=device,
    )
    rh_kalman = HandPoseKalmanFilter(
        fps=FPS, process_var=1e-4, obs_var=1e-2,
        confidence_power=2.0, outlier_threshold=0.8,
        outlier_noise_scale=10.0, device=device,
    )

    all_l_kalman, all_r_kalman = [], []
    for i in range(T):
        all_l_kalman.append(lh_kalman.update(all_l_raw[i], all_l_conf[i]))
        all_r_kalman.append(rh_kalman.update(all_r_raw[i], all_r_conf[i]))

    # 3) Temporal Smoother
    print("[INFO] Temporal 모델 처리 중...")
    lh_smoother = TemporalSmoother(TEMPORAL_MODEL_PATH, seq_len=31, device=device)
    rh_smoother = TemporalSmoother(TEMPORAL_MODEL_PATH, seq_len=31, device=device)

    all_l_temporal = lh_smoother.process_video(all_l_kalman)
    all_r_temporal = rh_smoother.process_video(all_r_kalman)

    return (
        all_l_raw,      all_r_raw,
        all_l_kalman,   all_r_kalman,
        all_l_temporal, all_r_temporal,
    )


# ──────────────────────────────────────────────────────────
# 애니메이션
# ──────────────────────────────────────────────────────────

# 모드 순환: raw → kalman → temporal → raw → ...
MODES       = ["raw", "kalman", "temporal"]
MODE_LABELS = {
    "raw":      "Raw (필터 없음)",
    "kalman":   "Kalman Filter",
    "temporal": "Temporal Smoothing",
}


def animate_motion(model, motion, fps=30):
    T = motion["body_pose"].shape[0]

    (all_l_raw, all_r_raw,
     all_l_kalman, all_r_kalman,
     all_l_temporal, all_r_temporal) = preprocess_hands(T)

    # 프레임 수를 가장 짧은 시퀀스에 맞춤
    T = min(T,
            len(all_l_raw), len(all_r_raw),
            len(all_l_kalman), len(all_r_kalman),
            len(all_l_temporal), len(all_r_temporal))
    print(f"[INFO] 재생 프레임 수: {T}")

    # ── 데이터 맵 ──────────────────────────────────────────
    hand_data = {
        "raw":      (all_l_raw,      all_r_raw),
        "kalman":   (all_l_kalman,   all_r_kalman),
        "temporal": (all_l_temporal, all_r_temporal),
    }

    # ── 재생 상태 ──────────────────────────────────────────
    state = {
        "paused":  False,
        "restart": False,
        "frame":   0,
        "mode":    "raw",       # raw | kalman | temporal
    }

    def _get_hands(i):
        l_list, r_list = hand_data[state["mode"]]
        return l_list[i], r_list[i]

    # ── 키 콜백 ────────────────────────────────────────────
    def toggle_pause(vis):
        state["paused"] = not state["paused"]
        print("[SPACE]", "일시정지" if state["paused"] else "재생")
        return False

    def cycle_mode(vis):
        idx          = MODES.index(state["mode"])
        state["mode"] = MODES[(idx + 1) % len(MODES)]
        print(f"[M] 모드 전환 → {MODE_LABELS[state['mode']]}")
        return False

    def toggle_kalman(vis):
        state["mode"] = "kalman" if state["mode"] != "kalman" else "raw"
        print(f"[K] {MODE_LABELS[state['mode']]}")
        return False

    def toggle_temporal(vis):
        state["mode"] = "temporal" if state["mode"] != "temporal" else "raw"
        print(f"[T] {MODE_LABELS[state['mode']]}")
        return False

    def restart(vis):
        state["restart"] = True
        state["paused"]  = False
        print("[R] 처음부터 재생")
        return False

    def step_forward(vis):
        state["paused"] = True
        state["frame"]  = min(state["frame"] + 1, T - 1)
        print(f"[→] 프레임: {state['frame']}")
        return False

    def step_backward(vis):
        state["paused"] = True
        state["frame"]  = max(state["frame"] - 1, 0)
        print(f"[←] 프레임: {state['frame']}")
        return False

    # ── 초기 메쉬 ──────────────────────────────────────────
    l0, r0, _, _ = load_hamer_hands(0)
    verts, faces, _ = make_mesh_from_frame(
        model,
        motion["body_pose"][0], motion["betas"][0],
        motion["global_orient"][0], motion["transl"][0],
        l0, r0,
    )
    mesh = create_open3d_mesh(verts, faces)

    # ── Visualizer ─────────────────────────────────────────
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name=(
            "HaMeR + GVHMR  |  SPACE: 정지/재생  |  R: 처음부터  "
            "|  ←→: 프레임  |  K: Kalman  |  T: Temporal  |  M: 모드 순환"
        ),
        width=1280, height=720,
    )

    vis.register_key_callback(ord(" "), toggle_pause)
    vis.register_key_callback(ord("R"), restart)
    vis.register_key_callback(ord("K"), toggle_kalman)
    vis.register_key_callback(ord("T"), toggle_temporal)
    vis.register_key_callback(ord("M"), cycle_mode)
    vis.register_key_callback(262, step_forward)
    vis.register_key_callback(263, step_backward)

    for elem in create_studio_room():
        vis.add_geometry(elem)
    vis.add_geometry(mesh)

    vis.get_render_option().background_color = np.array([0.1, 0.1, 0.1])

    frame_time = 1.0 / fps
    i          = 0

    print(f"\n[INFO] 재생 시작 ({T} 프레임)")
    print("  SPACE: 정지/재생  |  R: 처음부터  |  ←→: 프레임 이동")
    print("  K: Kalman  |  T: Temporal  |  M: 모드 순환 (raw → kalman → temporal)\n")

    try:
        while True:
            if state["restart"]:
                state["restart"] = False
                state["frame"]   = 0
                i                = 0

            if state["paused"]:
                i = state["frame"]
            else:
                state["frame"] = i

            l_hand, r_hand = _get_hands(i)

            start = time.time()
            verts, _, _ = make_mesh_from_frame(
                model,
                motion["body_pose"][i], motion["betas"][i],
                motion["global_orient"][i], motion["transl"][i],
                l_hand, r_hand,
            )
            mesh.vertices = o3d.utility.Vector3dVector(verts)
            mesh.compute_vertex_normals()
            vis.update_geometry(mesh)
            vis.poll_events()
            vis.update_renderer()

            if state["paused"]:
                time.sleep(0.03)
                continue

            elapsed = time.time() - start
            time.sleep(max(0.0, frame_time - elapsed))
            i = (i + 1) % T

    except KeyboardInterrupt:
        print("[INFO] 종료.")
    finally:
        vis.destroy_window()


# ──────────────────────────────────────────────────────────
# 진입점
# ──────────────────────────────────────────────────────────

def main():
    motion = load_motion(PT_PATH, PARAM_KEY)
    model  = build_smpl_model(MODEL_PATH, GENDER)
    animate_motion(model, motion, FPS)


if __name__ == "__main__":
    main()