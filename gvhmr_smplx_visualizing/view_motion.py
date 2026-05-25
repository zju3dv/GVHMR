# -*- coding: utf-8 -*-  
from __future__ import print_function

import time
import numpy as np
import torch
import open3d as o3d
import smplx
import os
import cv2

# --- 경로 및 설정 ---
HAMER_PATH = "C:\\Users\\SeunghyunWoo\\Desktop\\HaMeR_output\\hamer\\frames_out"
PT_PATH = "hmr4d_results.pt"
MODEL_PATH = "models"
PARAM_KEY = "smpl_params_global"
GENDER = "neutral"
FPS = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_motion(pt_path, param_key="smpl_params_global"):
    data = torch.load(pt_path, map_location=device)
    params = data[param_key]
    
    return {
        "body_pose": params["body_pose"].float(),
        "betas": params["betas"].float(),
        "global_orient": params["global_orient"].float(),
        "transl": params["transl"].float(),
    }

def build_smpl_model(model_path, gender="neutral"):
    # HaMeR는 보통 PCA가 아닌 Full Pose(45차원)를 주므로 use_pca=False 설정이 안전합니다.
    model = smplx.create(
        model_path=model_path,
        model_type="smplx",
        gender=gender,
        use_pca=False, 
        batch_size=1,
    ).to(device)
    return model

def load_hamer_hands(frame_idx):
    """OpenCV를 사용하여 HaMeR의 3x3 행렬을 SMPL-X용 데이터로 변환"""
    prefix = "{:04d}_".format(frame_idx)
    hand_files = [f for f in os.listdir(HAMER_PATH) if f.startswith(prefix) and f.endswith(".npz")]
    
    l_hand, r_hand = None, None
    
    for f in hand_files:
        try:
            data = np.load(os.path.join(HAMER_PATH, f), allow_pickle=True)
            is_right = data['is_right']
            
            # 1. HaMeR 데이터 추출 (보통 15, 3, 3 형태의 넘파이 배열)
            hand_pose_matrix = data['mano_params'].item()['hand_pose']
            
            # 2. [핵심] 15개 관절 각각을 3x3 행렬 -> 3차원 벡터로 변환
            aa_list = []
            for j in range(15):
                # cv2.Rodrigues는 (3,3) 행렬을 받아서 (3,1) 벡터와 야코비안을 반환합니다.
                R = hand_pose_matrix[j]
                aa, _ = cv2.Rodrigues(R)
                aa_list.append(aa.flatten()) # (3,) 형태로 펴서 저장
            
            # 3. (15, 3) -> (1, 45) 형태의 텐서로 변환
            hand_pose_final = torch.tensor(np.array(aa_list)).float().reshape(1, 45).to(device)

            if is_right == 1:
                r_hand = hand_pose_final
            else:
                l_hand = hand_pose_final
                
        except Exception as e:
            print(f"[WARN] {f} 로드 실패: {e}")
            
    return l_hand, r_hand

def create_racket_mesh():
    """간단한 라켓 메쉬 생성 (손잡이 + 헤드)"""
    # 1. 손잡이 (Handle)
    handle = o3d.geometry.TriangleMesh.create_cylinder(radius=0.015, height=0.4)
    handle.paint_uniform_color([0.5, 0.3, 0.1]) # 갈색(나무)
    
    # 2. 헤드 (Head) - 원형 프레임
    head = o3d.geometry.TriangleMesh.create_torus(torus_radius=0.15, tube_radius=0.01)
    head.translate([0, 0.35, 0]) # 손잡이 위로 이동
    head.paint_uniform_color([0.1, 0.1, 0.1]) # 검정색
    
    # 두 메쉬 합치기
    racket = handle + head
    racket.compute_vertex_normals()
    return racket

def make_mesh_from_frame(model, body_pose, betas, global_orient, transl, left_hand_pose=None, right_hand_pose=None):
    """배치 사이즈 1 고정 및 필수 포즈 명시적 전달"""
    with torch.no_grad():
        # 모든 입력을 (1, N) 형태로 리셰이프 (안전장치)
        b_pose = body_pose.reshape(1, -1)
        beta   = betas.reshape(1, -1)
        g_ori  = global_orient.reshape(1, -1)
        trans  = transl.reshape(1, -1)

        if left_hand_pose is None:
            left_hand_pose = torch.zeros(1, 45).to(device)
        if right_hand_pose is None:
            right_hand_pose = torch.zeros(1, 45).to(device)

        # SMPL-X 내부에서 Batch Size가 꼬이지 않도록 턱과 눈 포즈도 1개분(1, 3)으로 전달
        zero_pose = torch.zeros(1, 3).to(device)

        output = model(
            betas=betas.reshape(1, -1),
            global_orient=global_orient.reshape(1, -1),
            body_pose=body_pose.reshape(1, -1),
            transl=transl.reshape(1, -1),
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            return_verts=True,
            return_full_pose=True # 관절 각도를 가져오기 위해 추가
        )

    vertices = output.vertices[0].cpu().numpy()
    faces = model.faces.astype(np.int32)
    
    # [핵심] 오른손 손목(Joint 21)의 위치와 회전 정보를 가져옵니다.
    # SMPL-X 관절 인덱스: 21번이 오른손 손목입니다.
    right_wrist_pos = output.joints[0, 21].cpu().numpy()
    
    return vertices, faces, right_wrist_pos

# --- 시각화 관련 유틸리티 (기존과 동일) ---
def create_open3d_mesh(vertices, faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    return mesh

def create_studio_room():
    geometries = []
    floor = o3d.geometry.TriangleMesh.create_box(width=10, height=0.01, depth=10)
    floor.translate([-5, -1.0, -5])
    floor.paint_uniform_color([0.3, 0.3, 0.3])
    geometries.append(floor)
    
    back_wall = o3d.geometry.TriangleMesh.create_box(width=10, height=5, depth=0.01)
    back_wall.translate([-5, -1.0, -5])
    back_wall.paint_uniform_color([0.2, 0.2, 0.2])
    geometries.append(back_wall)
    
    return geometries

def animate_motion(model, motion, fps=30):
    T = motion["body_pose"].shape[0]

    # 1. 초기 셋팅
    l_hand, r_hand = load_hamer_hands(0)
    vertices, faces, racket_pos = make_mesh_from_frame(
        model, motion["body_pose"][0], motion["betas"][0], 
        motion["global_orient"][0], motion["transl"][0],
        l_hand, r_hand
    )


    mesh = create_open3d_mesh(vertices, faces)
    racket = create_racket_mesh() # 라켓 생성
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="HaMeR + GVHMR 통합 애니메이션", width=1280, height=720)
    
    # 배경 및 캐릭터 추가
    for element in create_studio_room(): vis.add_geometry(element)
    vis.add_geometry(mesh)
    # vis.add_geometry(racket) # 라켓 추가
    
    
    render_opt = vis.get_render_option()
    render_opt.background_color = np.asarray([0.1, 0.1, 0.1])
    
    print("[INFO] Start animation: {} frames".format(T)) 
    frame_time = 1.0 / fps

    try:
        while True:
            for i in range(T):
                start = time.time()
                l_hand, r_hand = load_hamer_hands(i)

                # 2. 메쉬와 라켓 위치 업데이트
                vertices, _, racket_pos = make_mesh_from_frame(
                    model, motion["body_pose"][i], motion["betas"][i],
                    motion["global_orient"][i], motion["transl"][i],
                    l_hand, r_hand
                )

                # 캐릭터 업데이트
                mesh.vertices = o3d.utility.Vector3dVector(vertices)
                mesh.compute_vertex_normals()
                
                # 라켓 업데이트 (손목 위치로 이동)
                # 손가락 각도에 맞춘 회전까지 넣으려면 Transformation Matrix를 써야 하지만,
                # 일단 위치만 맞춰도 '쥐고 있는' 느낌이 납니다!
                # racket.transform(np.eye(4)) # 초기화 느낌으로 갱신 (Open3D 특성상)
                # racket.translate(racket_pos, relative=False) 

                vis.update_geometry(mesh)
                vis.update_geometry(racket)
                vis.poll_events()
                vis.update_renderer()
                
                elapsed = time.time() - start
                time.sleep(max(0.0, frame_time - elapsed))

    except KeyboardInterrupt:
        print("[INFO] Stopped.")
    finally:
        vis.destroy_window()

def main():
    motion = load_motion(PT_PATH, PARAM_KEY)
    model = build_smpl_model(MODEL_PATH, GENDER)
    animate_motion(model, motion, FPS)

if __name__ == "__main__":
    main()