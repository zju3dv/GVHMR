import torch
import pytorch_lightning as pl
import numpy as np
from pathlib import Path
from einops import einsum, rearrange

from hmr4d.configs import MainStore, builds
from hmr4d.utils.pylogger import Log
from hmr4d.utils.comm.gather import all_gather
from hmr4d.utils.eval.eval_utils import compute_camcoord_metrics, as_np_array
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.vis.cv2_utils import cv2, draw_bbx_xys_on_image_batch, draw_coco17_skeleton_batch
from hmr4d.utils.vis.renderer_utils import simple_render_mesh_background
from hmr4d.utils.video_io_utils import read_video_np, get_video_lwh, save_video
from hmr4d.utils.geo_transform import apply_T_on_points
from hmr4d.utils.seq_utils import rearrange_by_mask


class MetricMocap(pl.Callback):
    def __init__(self):
        super().__init__()
        # vid->result
        self.metric_aggregator = {
            "pa_mpjpe": {},
            "mpjpe": {},
            "pve": {},
            "accel": {},
        }

        # SMPLX and SMPL
        self.smplx = make_smplx("supermotion_EVAL3DPW")
        self.smpl = {"male": make_smplx("smpl", gender="male"), "female": make_smplx("smpl", gender="female")}
        self.J_regressor = torch.load("hmr4d/utils/body_model/smpl_3dpw14_J_regressor_sparse.pt").to_dense()
        self.J_regressor24 = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt")
        self.smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt")
        self.faces_smplx = self.smplx.faces
        self.faces_smpl = self.smpl["male"].faces

        # The metrics are calculated similarly for val/test/predict
        self.on_test_batch_end = self.on_validation_batch_end = self.on_predict_batch_end

        # Only validation record the metrics with logger
        self.on_test_epoch_end = self.on_validation_epoch_end = self.on_predict_epoch_end

    # ================== Batch-based Computation  ================== #
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """The behaviour is the same for val/test/predict"""
        assert batch["B"] == 1
        dataset_id = batch["meta"][0]["dataset_id"]
        if dataset_id != "3DPW":
            return

        # Move to cuda if not
        self.smplx = self.smplx.cuda()
        for g in ["male", "female"]:
            self.smpl[g] = self.smpl[g].cuda()
        self.J_regressor = self.J_regressor.cuda()
        self.J_regressor24 = self.J_regressor24.cuda()
        self.smplx2smpl = self.smplx2smpl.cuda()

        vid = batch["meta"][0]["vid"]
        seq_length = batch["length"][0].item()
        gender = batch["gender"][0]
        T_w2c = batch["T_w2c"][0]
        mask = batch["mask"][0]

        # Groundtruth (cam)
        target_w_params = {k: v[0] for k, v in batch["smpl_params"].items()}
        target_w_output = self.smpl[gender](**target_w_params)
        target_w_verts = target_w_output.vertices
        target_c_verts = apply_T_on_points(target_w_verts, T_w2c)
        target_c_j3d = torch.matmul(self.J_regressor, target_c_verts)

        # + Prediction -> Metric
        smpl_out = self.smplx(**outputs["pred_smpl_params_incam"])
        pred_c_verts = torch.stack([torch.matmul(self.smplx2smpl, v_) for v_ in smpl_out.vertices])
        pred_c_j3d = einsum(self.J_regressor, pred_c_verts, "j v, l v i -> l j i")
        del smpl_out  # Prevent OOM

        # Metric of current sequence
        batch_eval = {
            "pred_j3d": pred_c_j3d,
            "target_j3d": target_c_j3d,
            "pred_verts": pred_c_verts,
            "target_verts": target_c_verts,
        }
        camcoord_metrics = compute_camcoord_metrics(batch_eval, mask=mask, pelvis_idxs=[2, 3])
        for k in camcoord_metrics:
            self.metric_aggregator[k][vid] = as_np_array(camcoord_metrics[k])

        if False:  # Render incam (simple)
            meta_render = batch["meta_render"][0]
            images = read_video_np(meta_render["video_path"], scale=meta_render["ds"])
            render_dict = {
                "K": meta_render["K"][None],  # only support batch size 1
                "faces": self.smpl["male"].faces,
                "verts": pred_c_verts,
                "background": images,
            }
            img_overlay = simple_render_mesh_background(render_dict)
            output_fn = Path("outputs/3DPW_render_pred_flip") / f"{vid}.mp4"
            save_video(img_overlay, output_fn, crf=28)

        if False:  # Render incam (with details)
            meta_render = batch["meta_render"][0]
            images = read_video_np(meta_render["video_path"], scale=meta_render["ds"])
            render_dict = {
                "K": meta_render["K"][None],  # only support batch size 1
                "faces": self.smpl["male"].faces,
                "verts": pred_c_verts,
                "background": images,
            }
            img_overlay = simple_render_mesh_background(render_dict)

            # Add COCO17 and bbx to image
            bbx_xys_render = meta_render["bbx_xys"]
            kp2d_render = meta_render["kp2d"]
            img_overlay = draw_coco17_skeleton_batch(img_overlay, kp2d_render, conf_thr=0.5)
            img_overlay = draw_bbx_xys_on_image_batch(bbx_xys_render, img_overlay, mask)

            # Add metric
            metric_all = rearrange_by_mask(torch.tensor(camcoord_metrics["pa_mpjpe"]), mask)
            for i in range(len(img_overlay)):
                m = metric_all[i]
                if m == 0:  # a not evaluated frame
                    continue
                text = f"PA-MPJPE: {m:.1f}"
                color = (244, 10, 20) if m > 45 else (0, 205, 0)  # red or green
                cv2.putText(img_overlay[i], text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            output_dir = Path("tmp_pred_details")
            output_dir.mkdir(exist_ok=True, parents=True)
            save_video(img_overlay, output_dir / f"{vid}.mp4", crf=24)

    # ================== Epoch Summary  ================== #
    def on_predict_epoch_end(self, trainer, pl_module):
        """Without logger"""
        local_rank, world_size = trainer.local_rank, trainer.world_size
        monitor_metric = "pa_mpjpe"

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
            Log.info(f"[Metrics] 3DPW:\n" + "\n".join(f"{k}: {v:.1f}" for k, v in metrics_avg.items()) + "\n------")

        # save to logger if available
        if pl_module.logger is not None:
            cur_epoch = pl_module.current_epoch
            for k, v in metrics_avg.items():
                pl_module.logger.log_metrics({f"val_metric_3DPW/{k}": v}, step=cur_epoch)

        # reset
        for k in self.metric_aggregator:
            self.metric_aggregator[k] = {}


node_3dpw = builds(MetricMocap)
MainStore.store(name="metric_3dpw", node=node_3dpw, group="callbacks", package="callbacks.metric_3dpw")
