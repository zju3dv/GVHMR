import os
os.environ['PYOPENGL_PLATFORM'] = 'pyglet'

from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

from vitpose_model import ViTPoseModel

import json
from typing import Dict, Optional

def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')
    parser.add_argument('--no_render', dest='no_render', action='store_true', default=False, help='If set, will not run the renderer at all')

    args = parser.parse_args()

    # Download and load checkpoints
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif args.body_detector == 'regnety':
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
        detector       = DefaultPredictor_Lazy(detectron2_cfg)

    # keypoint detector
    cpm = ViTPoseModel(device)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    # Get all demo images ends with .jpg or .png
    img_paths = [img for end in args.file_type for img in Path(args.img_folder).glob(end)]

    # Iterate over all images in folder
    for img_path in img_paths:
        img_cv2 = cv2.imread(str(img_path))

        # Detect humans in image
        det_out = detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.8)
        pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores=det_instances.scores[valid_idx].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = cpm.predict_pose(
            img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bboxes = []
        is_right = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Rejecting not confident detections
            keyp = left_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(0)
            keyp = right_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(1)

        if len(bboxes) == 0:
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        # [수정] 왼손, 오른손 각각에서 면적이 가장 큰 바운딩 박스를 추출
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        final_indices = []

        # 1. 오른손(right == 1) 중 가장 큰 것 찾기
        right_indices = np.where(right == 1)[0]
        if len(right_indices) > 0:
            largest_right_idx = right_indices[np.argmax(areas[right_indices])]
            final_indices.append(largest_right_idx)

        # 2. 왼손(right == 0) 중 가장 큰 것 찾기
        left_indices = np.where(right == 0)[0]
        if len(left_indices) > 0:
            largest_left_idx = left_indices[np.argmax(areas[left_indices])]
            final_indices.append(largest_left_idx)

        # 최종 선택된 인덱스들만 남기기 (최대 2개: 왼손1, 오른손1)
        if len(final_indices) > 0:
            boxes = boxes[final_indices]
            right = right[final_indices]
        else:
            continue # 검출된 손이 없으면 다음 이미지로
        
        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
        # [핵심] 바운딩 박스 시각화 및 저장 로직 추가
        img_bbox = img_cv2.copy()
        for box, is_right_hand in zip(boxes, right):
            color = (0, 255, 0) if is_right_hand else (255, 0, 0) # 오른손 녹색, 왼손 적색
            cv2.rectangle(img_bbox, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 3)
        
        # 원본 이미지 이름 뒤에 _bbox를 붙여서 저장
        img_fn, _ = os.path.splitext(os.path.basename(img_path))
        cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_bbox.jpg'), img_bbox)

        all_verts = []
        all_cam_t = []
        all_right = []
        
        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            multiplier = (2*batch['right']-1)
            pred_cam = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            multiplier = (2*batch['right']-1)
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                # Get filename from path img_path
                img_fn, _ = os.path.splitext(os.path.basename(img_path))
                person_id = int(batch['personid'][n])
                
                # [핵심] 3D 수치 데이터(.npz) 저장 로직 추가
                out_npz_path = os.path.join(args.out_folder, f'{img_fn}_{person_id}.npz')
                np.savez(out_npz_path,
                        vertices=out['pred_vertices'][n].detach().cpu().numpy(),
                        cam_t=out['pred_cam_t'][n].detach().cpu().numpy(),
                        mano_params={k: v[n].detach().cpu().numpy() for k, v in out['pred_mano_params'].items()},
                        is_right=batch['right'][n].cpu().numpy())
                
                # 렌더링 옵션이 켜져 있을 때만 실행 (에러 방지)
                if not args.no_render:
                    try:
                        # 입력 패치 준비
                        input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                        input_patch = input_patch.permute(1,2,0).numpy()

                        # 메인 렌더링
                        regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                                out['pred_cam_t'][n].detach().cpu().numpy(),
                                                batch['img'][n],
                                                mesh_base_color=LIGHT_BLUE,
                                                scene_bg_color=(1, 1, 1))
                        
                        if args.side_view:
                            white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
                            side_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                                out['pred_cam_t'][n].detach().cpu().numpy(),
                                                white_img,
                                                mesh_base_color=LIGHT_BLUE,
                                                scene_bg_color=(1, 1, 1),
                                                side_view=True)
                            final_img = np.concatenate([input_patch, regression_img, side_img], axis=1)
                        else:
                            final_img = np.concatenate([input_patch, regression_img], axis=1)

                        cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_{person_id}.png'), 255*final_img[:, :, ::-1])
                    
                    except Exception as e:
                        print(f"⚠️ {img_fn}_{person_id} 렌더링 실패 (데이터는 보존됨): {e}")

                # Add all verts and cams to list
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                is_right = batch['right'][n].cpu().numpy()
                verts[:,0] = (2*is_right-1)*verts[:,0]
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)

                # Save all meshes to disk
                if args.save_mesh:
                    camera_translation = cam_t.copy()
                    tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE, is_right=is_right)
                    tmesh.export(os.path.join(args.out_folder, f'{img_fn}_{person_id}.obj'))

        # Render front view
        if args.full_frame and len(all_verts) > 0 and not args.no_render:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            try:
                cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)
            except Exception as e:
                print(f"⚠️ 전체 뷰 렌더링 실패 (무시하고 저장): {e}")
                cam_view = np.zeros((img_size[n][1], img_size[n][0], 3)) # 빈 화면으로 대체
                
            # Overlay image
            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

            cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_all.jpg'), 255*input_img_overlay[:, :, ::-1])

if __name__ == '__main__':
    main()
