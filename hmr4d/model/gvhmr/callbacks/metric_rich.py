import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from hmr4d.configs import MainStore, builds

from hmr4d.utils.comm.gather import all_gather
from hmr4d.utils.pylogger import Log

from hmr4d.utils.eval.eval_utils import (
    compute_camcoord_metrics,
    compute_global_metrics,
    compute_camcoord_perjoint_metrics,
    as_np_array,
)
from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from hmr4d.utils.smplx_utils import make_smplx
from einops import einsum, rearrange

from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines, get_colors_by_conf
from hmr4d.utils.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points
from hmr4d.utils.geo.hmr_cam import estimate_focal_length
from hmr4d.utils.video_io_utils import read_video_np, save_video, get_writer
import imageio
from tqdm import tqdm
from pathlib import Path
import numpy as np
import cv2

from smplx.joint_names import JOINT_NAMES
from hmr4d.utils.net_utils import repeat_to_max_len, gaussian_smooth
from hmr4d.utils.geo.hmr_global import rollout_vel, get_static_joint_mask


class MetricMocap(pl.Callback):
    def __init__(self):
        super().__init__()
        # vid->result
        self.metric_aggregator = {
            "pa_mpjpe": {},
            "mpjpe": {},
            "pve": {},
            "accel": {},
            "wa2_mpjpe": {},
            "waa_mpjpe": {},
            "rte": {},
            "jitter": {},
            "fs": {},
        }

        self.perjoint_metrics = False
        if self.perjoint_metrics:
            body_joint_names = JOINT_NAMES[:22] + ["left_hand", "right_hand"]
            self.body_joint_names = body_joint_names
            self.perjoint_metric_aggregator = {
                "mpjpe": {k: {} for k in body_joint_names},
            }
            self.perjoint_obs_metric_aggregator = {
                "mpjpe": {k: {} for k in body_joint_names},
            }

        # SMPL
        self.smplx_model = {
            "male": make_smplx("rich-smplx", gender="male"),
            "female": make_smplx("rich-smplx", gender="female"),
            "neutral": make_smplx("rich-smplx", gender="neutral"),
        }
        self.J_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt")
        self.smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt")
        self.faces_smpl = make_smplx("smpl").faces
        self.faces_smplx = self.smplx_model["neutral"].faces

        # The metrics are calculated similarly for val/test/predict
        self.on_test_batch_end = self.on_validation_batch_end = self.on_predict_batch_end

        # Only validation record the metrics with logger
        self.on_test_epoch_end = self.on_validation_epoch_end = self.on_predict_epoch_end

    # ================== Batch-based Computation  ================== #
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """The behaviour is the same for val/test/predict"""
        assert batch["B"] == 1
        dataset_id = batch["meta"][0]["dataset_id"]
        if dataset_id != "RICH":
            return

        # Move to cuda if not
        for g in ["male", "female", "neutral"]:
            self.smplx_model[g] = self.smplx_model[g].cuda()
        self.J_regressor = self.J_regressor.cuda()
        self.smplx2smpl = self.smplx2smpl.cuda()

        vid = batch["meta"][0]["vid"]
        seq_length = batch["length"][0].item()
        gender = batch["gender"][0]
        T_w2ay = batch["T_w2ay"][0]
        T_w2c = batch["T_w2c"][0]

        # Groundtruth (world, cam)
        target_w_params = {k: v[0] for k, v in batch["gt_smpl_params"].items()}
        target_w_output = self.smplx_model[gender](**target_w_params)
        target_w_verts = torch.stack([torch.matmul(self.smplx2smpl, v_) for v_ in target_w_output.vertices])
        target_c_verts = apply_T_on_points(target_w_verts, T_w2c)
        target_c_j3d = torch.matmul(self.J_regressor, target_c_verts)
        offset = target_c_j3d[..., [1, 2], :].mean(-2, keepdim=True)  # (L, 1, 3)
        target_cr_j3d = target_c_j3d - offset
        target_cr_verts = target_c_verts - offset
        # optional: ay for visual comparison
        target_ay_verts = apply_T_on_points(target_w_verts, T_w2ay)
        target_ay_j3d = torch.matmul(self.J_regressor, target_ay_verts)

        # + Prediction -> Metric
        # 1. cam
        pred_smpl_params_incam = outputs["pred_smpl_params_incam"]
        smpl_out = self.smplx_model["neutral"](**pred_smpl_params_incam)
        pred_c_verts = torch.stack([torch.matmul(self.smplx2smpl, v_) for v_ in smpl_out.vertices])
        pred_c_j3d = einsum(self.J_regressor, pred_c_verts, "j v, l v i -> l j i")
        offset = pred_c_j3d[..., [1, 2], :].mean(-2, keepdim=True)  # (L, 1, 3)

        # 2. ay
        pred_smpl_params_global = outputs["pred_smpl_params_global"]
        smpl_out = self.smplx_model["neutral"](**pred_smpl_params_global)
        pred_ay_verts = torch.stack([torch.matmul(self.smplx2smpl, v_) for v_ in smpl_out.vertices])
        pred_ay_j3d = einsum(self.J_regressor, pred_ay_verts, "j v, l v i -> l j i")

        # Metric of current sequence
        batch_eval = {
            "pred_j3d": pred_c_j3d,
            "target_j3d": target_c_j3d,
            "pred_verts": pred_c_verts,
            "target_verts": target_c_verts,
        }
        camcoord_metrics = compute_camcoord_metrics(batch_eval)
        for k in camcoord_metrics:
            self.metric_aggregator[k][vid] = as_np_array(camcoord_metrics[k])

        batch_eval = {
            "pred_j3d_glob": pred_ay_j3d,
            "target_j3d_glob": target_ay_j3d,
            "pred_verts_glob": pred_ay_verts,
            "target_verts_glob": target_ay_verts,
        }
        global_metrics = compute_global_metrics(batch_eval)
        for k in global_metrics:
            self.metric_aggregator[k][vid] = as_np_array(global_metrics[k])

        if False:  # global wi3d debug
            wis3d = make_wis3d(name="debug-metric-global")
            add_motion_as_lines(pred_ay_j3d, wis3d, name="pred_ay_j3d")
            add_motion_as_lines(target_ay_j3d, wis3d, name="target_ay_j3d")

        if False:  # incam visualize debug
            # Print per-sequence error
            Log.info(
                f"seq {vid} metrics:\n"
                + "\n".join(
                    f"{k}: {self.metric_aggregator[k][vid].mean():.1f} (obs:{camcoord_metrics[k].mean():.1f})"
                    for k in camcoord_metrics.keys()
                )
                + "\n------\n"
            )
            if self.perjoint_metrics:
                Log.info(
                    f"\n".join(
                        f"{k}-{j}: {self.perjoint_metric_aggregator[k][j][vid].mean():.1f} (obs:{self.perjoint_obs_metric_aggregator[k][j][vid].mean():.1f})"
                        for j in self.body_joint_names
                        for k in self.perjoint_obs_metric_aggregator.keys()
                    )
                    + "\n------"
                )

            # -- metric -- #
            pred_mpjpe = self.metric_aggregator["mpjpe"][vid].mean()
            obs_mpjpe = camcoord_metrics["mpjpe"].mean()

            # -- render mesh -- #
            vertices_gt = target_c_verts
            vertices_cr_gt = target_cr_verts + target_cr_verts.new([0, 0, 3.0])  # move forward +z
            vertices_pred = pred_c_verts
            vertices_cr_obs = obs_cr_verts + obs_cr_verts.new([0, 0, 3.0])  # move forward +z
            vertices_cr_pred = pred_cr_verts + pred_cr_verts.new([0, 0, 3.0])  # move forward +z

            # -- rendering code -- #
            vname = batch["meta_render"][0]["name"]
            K = batch["meta_render"][0]["K"]
            width, height = batch["meta_render"][0]["width_height"]
            faces = self.faces_smpl

            renderer = Renderer(width, height, device="cuda", faces=faces, K=K)
            out_fn = f"outputs/dump_render/{vname}.mp4"
            Path(out_fn).parent.mkdir(exist_ok=True, parents=True)
            writer = imageio.get_writer(out_fn, fps=30, mode="I", format="FFMPEG", macro_block_size=1)

            # imgs
            video_path = batch["meta_render"][0]["video_path"]
            frame_id = batch["meta_render"][0]["frame_id"].cpu().numpy()
            vr = decord.VideoReader(video_path)
            images = vr.get_batch(list(frame_id)).numpy()  # (F, H/4, W/4, 3), uint8, numpy

            for i in tqdm(range(seq_length), desc=f"Rendering {vname}"):
                img_overlay_gt = renderer.render_mesh(vertices_gt[i].cuda(), images[i], [39, 194, 128])
                if batch["meta_render"][0].get("bbx_xys", None) is not None:  # draw bbox lines
                    bbx_xys = batch["meta_render"][0]["bbx_xys"][i].cpu().numpy()
                    lu_point = (bbx_xys[:2] - bbx_xys[2:] / 2).astype(int)
                    rd_point = (bbx_xys[:2] + bbx_xys[2:] / 2).astype(int)
                    img_overlay_gt = cv2.rectangle(img_overlay_gt, lu_point, rd_point, (255, 178, 102), 2)

                img_overlay_pred = renderer.render_mesh(vertices_pred[i].cuda(), images[i])
                # img_overlay_pred = renderer.render_mesh(vertices_pred[i].cuda(), np.zeros_like(images[i]))
                img = np.concatenate([img_overlay_gt, img_overlay_pred], axis=0)

                ####### overlay gt cr first, then overlay pred cr with error color ########
                # overlay gt cr first with blue color
                black_overlay_obs = renderer.render_mesh(
                    vertices_cr_gt[i].cuda(), np.zeros_like(images[i]), colors=[39, 194, 128]
                )
                black_overlay_pred = renderer.render_mesh(
                    vertices_cr_gt[i].cuda(), np.zeros_like(images[i]), colors=[39, 194, 128]
                )

                # get error color
                obs_error = (vertices_cr_gt[i] - vertices_cr_obs[i]).norm(dim=-1)
                pred_error = (vertices_cr_gt[i] - vertices_cr_pred[i]).norm(dim=-1)
                max_error = max(obs_error.max(), pred_error.max())
                obs_error_color = torch.stack(
                    [obs_error / max_error, torch.ones_like(obs_error) * 0.6, torch.ones_like(obs_error) * 0.6],
                    dim=-1,
                )
                obs_error_color = torch.clip(obs_error_color, 0, 1)
                pred_error_color = torch.stack(
                    [pred_error / max_error, torch.ones_like(pred_error) * 0.6, torch.ones_like(pred_error) * 0.6],
                    dim=-1,
                )
                pred_error_color = torch.clip(pred_error_color, 0, 1)

                # overlay cr with error color
                black_overlay_obs = renderer.render_mesh(
                    vertices_cr_obs[i].cuda(), black_overlay_obs, colors=obs_error_color[None]
                )
                black_overlay_pred = renderer.render_mesh(
                    vertices_cr_pred[i].cuda(), black_overlay_pred, colors=pred_error_color[None]
                )

                # write mpjpe on the img
                obs_mpjpe_ = camcoord_metrics["mpjpe"][i]
                text = f"obs mpjpe: {obs_mpjpe_:.1f} ({obs_mpjpe:.1f})"
                cv2.putText(black_overlay_obs, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 200), 2)
                pred_mpjpe_ = self.metric_aggregator["mpjpe"][vid][i]
                text = f"pred mpjpe: {pred_mpjpe_:.1f} ({pred_mpjpe:.1f})"
                if pred_mpjpe_ > obs_mpjpe_:
                    # large error -> purple
                    cv2.putText(black_overlay_pred, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 100, 200), 2)
                else:
                    # small error -> yellow
                    cv2.putText(black_overlay_pred, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 100), 2)
                black = np.concatenate([black_overlay_obs, black_overlay_pred], axis=0)
                ###########################################

                img = np.concatenate([img, black], axis=1)

                writer.append_data(img)
            writer.close()

        if False:  # Visualize incam + global results

            def move_to_start_point_face_z(verts):
                "XZ to origin, Start from the ground, Face-Z"
                # position
                verts = verts.clone()  # (L, V, 3)
                offset = einsum(self.J_regressor, verts[0], "j v, v i -> j i")[0]  # (3)
                offset[1] = verts[:, :, [1]].min()
                verts = verts - offset
                # face direction
                T_ay2ayfz = compute_T_ayfz2ay(einsum(self.J_regressor, verts[[0]], "j v, l v i -> l j i"), inverse=True)
                verts = apply_T_on_points(verts, T_ay2ayfz)
                return verts

            verts_incam = pred_c_verts.clone()
            # verts_glob = move_to_start_point_face_z(target_ay_verts)  # gt
            verts_glob = move_to_start_point_face_z(pred_ay_verts)
            joints_glob = einsum(self.J_regressor, verts_glob, "j v, l v i -> l j i")  # (L, J, 3)
            global_R, global_T, global_lights = get_global_cameras_static(
                verts_glob.cpu(),
                beta=4.0,
                cam_height_degree=20,
                target_center_height=1.0,
                vec_rot=-45,
            )

            # -- rendering code (global version FOV=55) -- #
            vname = batch["meta_render"][0]["name"]
            width, height = batch["meta_render"][0]["width_height"]
            K = batch["meta_render"][0]["K"]
            faces = self.faces_smpl
            out_fn = f"outputs/dump_render_global/{vname}.mp4"
            Path(out_fn).parent.mkdir(exist_ok=True, parents=True)

            # two renderers
            renderer_incam = Renderer(width, height, device="cuda", faces=faces, K=K)
            renderer_glob = Renderer(width, height, estimate_focal_length(width, height), device="cuda", faces=faces)

            # imgs
            video_path = batch["meta_render"][0]["video_path"]
            frame_id = batch["meta_render"][0]["frame_id"].cpu().numpy()
            images = read_video_np(video_path)[frame_id]  # (F, H/4, W/4, 3), uint8, numpy

            # Actual rendering
            scale, cx, cz = get_ground_params_from_points(joints_glob[:, 0], verts_glob)
            renderer_glob.set_ground(scale * 1.5, cx, cz)
            color = torch.ones(3).float().cuda() * 0.8

            writer = get_writer(out_fn, fps=30, crf=23)
            for i in tqdm(range(seq_length), desc=f"Rendering {vname}"):
                # incam
                img_overlay_pred = renderer_incam.render_mesh(verts_incam[i].cuda(), images[i], [0.8, 0.8, 0.8])
                # if batch["meta_render"][0].get("bbx_xys", None) is not None:  # draw bbox lines
                #     bbx_xys = batch["meta_render"][0]["bbx_xys"][i].cpu().numpy()
                #     lu_point = (bbx_xys[:2] - bbx_xys[2:] / 2).astype(int)
                #     rd_point = (bbx_xys[:2] + bbx_xys[2:] / 2).astype(int)
                #     img_overlay_pred = cv2.rectangle(img_overlay_pred, lu_point, rd_point, (255, 178, 102), 2)
                # pred_mpjpe_ = self.metric_aggregator["mpjpe"][vid][i]
                # text = f"pred mpjpe: {pred_mpjpe_:.1f}"
                # cv2.putText(img_overlay_pred, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 100, 200), 2)

                # glob
                cameras = renderer_glob.create_camera(global_R[i], global_T[i])
                # img_glob = renderer_glob.render_with_ground(verts_glob[[i]], color_[None], cameras, global_lights)
                img_glob = renderer_glob.render_with_ground(
                    verts_glob[[i]], color.clone()[None], cameras, global_lights
                )

                # write
                img = np.concatenate([img_overlay_pred, img_glob], axis=1)
                writer.write_frame(img)
            writer.close()

    # ================== Epoch Summary  ================== #
    def on_predict_epoch_end(self, trainer, pl_module):
        """Without logger"""
        local_rank, world_size = trainer.local_rank, trainer.world_size
        monitor_metric = "mpjpe"

        # Reduce metric_aggregator across all processes
        metric_keys = list(self.metric_aggregator.keys())
        with torch.inference_mode(False):  # allow in-place operation of all_gather
            metric_aggregator_gathered = all_gather(self.metric_aggregator)  # list of dict
        for metric_key in metric_keys:
            for d in metric_aggregator_gathered:
                self.metric_aggregator[metric_key].update(d[metric_key])

        if False:  # debug to make sure the all_gather is correct
            print(f"[RANK {local_rank}/{world_size}]: {self.metric_aggregator[monitor_metric].keys()}")

        total = len(self.metric_aggregator[monitor_metric])
        Log.info(f"{total} sequences evaluated in {self.__class__.__name__}")
        if total == 0:
            return

        # print monitored metric per sequence
        mm_per_seq = {k: v.mean() for k, v in self.metric_aggregator[monitor_metric].items()}
        if len(mm_per_seq) > 0:
            sorted_mm_per_seq = sorted(mm_per_seq.items(), key=lambda x: x[1], reverse=True)
            n_worst = 5 if trainer.state.stage == "validate" else len(sorted_mm_per_seq)
            if local_rank == 0:
                Log.info(
                    f"monitored metric {monitor_metric} per sequence\n"
                    + "\n".join([f"{m:5.1f} : {s}" for s, m in sorted_mm_per_seq[:n_worst]])
                    + "\n------"
                )

        # average over all batches
        metrics_avg = {k: np.concatenate(list(v.values())).mean() for k, v in self.metric_aggregator.items()}
        if local_rank == 0:
            Log.info(f"[Metrics] RICH:\n" + "\n".join(f"{k}: {v:.1f}" for k, v in metrics_avg.items()) + "\n------")

        # save to logger if available
        if pl_module.logger is not None:
            cur_epoch = pl_module.current_epoch
            for k, v in metrics_avg.items():
                pl_module.logger.log_metrics({f"val_metric_RICH/{k}": v}, step=cur_epoch)

        # reset
        for k in self.metric_aggregator:
            self.metric_aggregator[k] = {}


rich_node = builds(MetricMocap)
MainStore.store(name="metric_rich", node=rich_node, group="callbacks", package="callbacks.metric_rich")
