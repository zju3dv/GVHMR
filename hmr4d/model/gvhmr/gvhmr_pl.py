from typing import Any, Dict
import numpy as np
from pathlib import Path
import torch
import pytorch_lightning as pl
from hydra.utils import instantiate
from hmr4d.utils.pylogger import Log
from einops import rearrange, einsum
from hmr4d.configs import MainStore, builds

from hmr4d.utils.geo_transform import compute_T_ayfz2ay, apply_T_on_points
from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.geo.augment_noisy_pose import (
    get_wham_aug_kp3d,
    get_visible_mask,
    get_invisible_legs_mask,
    randomly_occlude_lower_half,
    randomly_modify_hands_legs,
)
from hmr4d.utils.geo.hmr_cam import perspective_projection, normalize_kp2d, safely_render_x3d_K, get_bbx_xys

from hmr4d.utils.video_io_utils import save_video
from hmr4d.utils.vis.cv2_utils import draw_bbx_xys_on_image_batch
from hmr4d.utils.geo.flip_utils import flip_smplx_params, avg_smplx_aa
from hmr4d.model.gvhmr.utils.postprocess import pp_static_joint, pp_static_joint_cam, process_ik


class GvhmrPL(pl.LightningModule):
    def __init__(
        self,
        pipeline,
        optimizer=None,
        scheduler_cfg=None,
        ignored_weights_prefix=["smplx", "pipeline.endecoder"],
    ):
        super().__init__()
        self.pipeline = instantiate(pipeline, _recursive_=False)
        self.optimizer = instantiate(optimizer)
        self.scheduler_cfg = scheduler_cfg

        # Options
        self.ignored_weights_prefix = ignored_weights_prefix

        # The test step is the same as validation
        self.test_step = self.predict_step = self.validation_step

        # SMPLX
        self.smplx = make_smplx("supermotion_v437coco17")

    def training_step(self, batch, batch_idx):
        B, F = batch["smpl_params_c"]["body_pose"].shape[:2]

        # Create augmented noisy-obs : gt_j3d(coco17)
        with torch.no_grad():
            gt_verts437, gt_j3d = self.smplx(**batch["smpl_params_c"])
            root_ = gt_j3d[:, :, [11, 12], :].mean(-2, keepdim=True)
            batch["gt_j3d"] = gt_j3d
            batch["gt_cr_coco17"] = gt_j3d - root_
            batch["gt_c_verts437"] = gt_verts437
            batch["gt_cr_verts437"] = gt_verts437 - root_

        # bbx_xys
        i_x2d = safely_render_x3d_K(gt_verts437, batch["K_fullimg"], thr=0.3)
        bbx_xys = get_bbx_xys(i_x2d, do_augment=True)
        if False:  # trust image bbx_xys seems better
            batch["bbx_xys"] = bbx_xys
        else:
            mask_bbx_xys = batch["mask"]["bbx_xys"]
            batch["bbx_xys"][~mask_bbx_xys] = bbx_xys[~mask_bbx_xys]
        if False:  # visualize bbx_xys from an iPhone view
            render_w, render_h = 120, 160  # iphone main-lens 24mm 3:4
            ratio = render_w / 1528
            offset = torch.tensor([764 - 500, 1019 - 500]).to(i_x2d)
            i_x2d_render = (i_x2d + offset).clone()
            i_x2d_render = (i_x2d_render * ratio).long().clone()
            torch.clamp_(i_x2d_render[..., 0], 0, render_w - 1)
            torch.clamp_(i_x2d_render[..., 1], 0, render_h - 1)
            bbx_xys_render = bbx_xys.clone()
            bbx_xys_render[..., :2] += offset
            bbx_xys_render *= ratio

            output_dir = Path("outputs/simulated_bbx_xys")
            output_dir.mkdir(parents=True, exist_ok=True)
            video_list = []
            for bid in range(B):
                images = torch.zeros(F, render_h, render_w, 3, device=i_x2d.device)
                for fid in range(F):
                    images[fid, i_x2d_render[bid, fid, :, 1], i_x2d_render[bid, fid, :, 0]] = 255

                images = draw_bbx_xys_on_image_batch(bbx_xys_render[bid].cpu().numpy(), images.cpu().numpy())
                images = np.stack(images).astype("uint8")  # (L, H, W, 3)
                images[:, 0, :] = np.array([255, 255, 255])
                images[:, -1, :] = np.array([255, 255, 255])
                images[:, :, 0] = np.array([255, 255, 255])
                images[:, :, -1] = np.array([255, 255, 255])
                video_list.append(images)

            # stack videos
            video_output = []
            for i in range(0, len(video_list), 4):
                if i + 4 <= len(video_list):
                    video_output.append(np.concatenate(video_list[i : i + 4], axis=2))
            video_output = np.concatenate(video_output, axis=1)
            save_video(video_output, output_dir / f"{batch_idx}.mp4", fps=30, quality=5)

        # noisy_j3d -> project to i_j2d -> compute a bbx -> normalized kp2d [-1, 1]
        noisy_j3d = gt_j3d + get_wham_aug_kp3d(gt_j3d.shape[:2])
        if True:
            noisy_j3d = randomly_modify_hands_legs(noisy_j3d)
        obs_i_j2d = perspective_projection(noisy_j3d, batch["K_fullimg"])  # (B, L, J, 2)
        j2d_visible_mask = get_visible_mask(gt_j3d.shape[:2]).cuda()  # (B, L, J)
        j2d_visible_mask[noisy_j3d[..., 2] < 0.3] = False  # Set close-to-image-plane points as invisible
        if True:  # Set both legs as invisible for a period
            legs_invisible_mask = get_invisible_legs_mask(gt_j3d.shape[:2]).cuda()  # (B, L, J)
            j2d_visible_mask[legs_invisible_mask] = False
        obs_kp2d = torch.cat([obs_i_j2d, j2d_visible_mask[:, :, :, None].float()], dim=-1)  # (B, L, J, 3)
        obs = normalize_kp2d(obs_kp2d, batch["bbx_xys"])  # (B, L, J, 3)
        obs[~j2d_visible_mask] = 0  # if not visible, set to (0,0,0)
        batch["obs"] = obs

        if True:  # Use some detected vitpose (presave data)
            prob = 0.5
            mask_real_vitpose = (torch.rand(B).to(obs_kp2d) < prob) * batch["mask"]["vitpose"]
            batch["obs"][mask_real_vitpose] = normalize_kp2d(batch["kp2d"], batch["bbx_xys"])[mask_real_vitpose]

        # Set untrusted frames to False
        batch["obs"][~batch["mask"]["valid"]] = 0

        if False:  # wis3d
            wis3d = make_wis3d(name="debug-aug-kp3d")
            add_motion_as_lines(gt_j3d[0], wis3d, name="gt_j3d", skeleton_type="coco17")
            add_motion_as_lines(noisy_j3d[0], wis3d, name="noisy_j3d", skeleton_type="coco17")

        # f_imgseq: apply random aug on offline extracted features
        # f_imgseq = batch["f_imgseq"] + torch.randn_like(batch["f_imgseq"]) * 0.1
        # f_imgseq[~batch["mask"]["f_imgseq"]] = 0
        # batch["f_imgseq"] = f_imgseq.clone()

        # Forward and get loss
        outputs = self.pipeline.forward(batch, train=True)

        # Log
        log_kwargs = {
            "on_epoch": True,
            "prog_bar": True,
            "logger": True,
            "batch_size": B,
            "sync_dist": True,
        }
        self.log("train/loss", outputs["loss"], **log_kwargs)
        for k, v in outputs.items():
            if "_loss" in k:
                self.log(f"train/{k}", v, **log_kwargs)

        return outputs

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Options & Check
        do_postproc = self.trainer.state.stage == "test"  # Only apply postproc in test
        do_flip_test = "flip_test" in batch
        do_postproc_not_flip_test = do_postproc and not do_flip_test  # later pp when flip_test
        assert batch["B"] == 1, "Only support batch size 1 in evalution."

        # ROPE inference
        obs = normalize_kp2d(batch["kp2d"], batch["bbx_xys"])
        if "mask" in batch:
            obs[0, ~batch["mask"][0]] = 0

        batch_ = {
            "length": batch["length"],
            "obs": obs,
            "bbx_xys": batch["bbx_xys"],
            "K_fullimg": batch["K_fullimg"],
            "cam_angvel": batch["cam_angvel"],
            "f_imgseq": batch["f_imgseq"],
        }
        outputs = self.pipeline.forward(batch_, train=False, postproc=do_postproc_not_flip_test)
        outputs["pred_smpl_params_global"] = {k: v[0] for k, v in outputs["pred_smpl_params_global"].items()}
        outputs["pred_smpl_params_incam"] = {k: v[0] for k, v in outputs["pred_smpl_params_incam"].items()}

        if do_flip_test:
            flip_test = batch["flip_test"]
            obs = normalize_kp2d(flip_test["kp2d"], flip_test["bbx_xys"])
            if "mask" in batch:
                obs[0, ~batch["mask"][0]] = 0

            batch_ = {
                "length": batch["length"],
                "obs": obs,
                "bbx_xys": flip_test["bbx_xys"],
                "K_fullimg": batch["K_fullimg"],
                "cam_angvel": flip_test["cam_angvel"],
                "f_imgseq": flip_test["f_imgseq"],
            }
            flipped_outputs = self.pipeline.forward(batch_, train=False)

            # First update incam results
            flipped_outputs["pred_smpl_params_incam"] = {
                k: v[0] for k, v in flipped_outputs["pred_smpl_params_incam"].items()
            }
            smpl_params1 = outputs["pred_smpl_params_incam"]
            smpl_params2 = flip_smplx_params(flipped_outputs["pred_smpl_params_incam"])

            smpl_params_avg = smpl_params1.copy()
            smpl_params_avg["betas"] = (smpl_params1["betas"] + smpl_params2["betas"]) / 2
            smpl_params_avg["body_pose"] = avg_smplx_aa(smpl_params1["body_pose"], smpl_params2["body_pose"])
            smpl_params_avg["global_orient"] = avg_smplx_aa(
                smpl_params1["global_orient"], smpl_params2["global_orient"]
            )
            outputs["pred_smpl_params_incam"] = smpl_params_avg

            # Then update global results
            outputs["pred_smpl_params_global"]["betas"] = smpl_params_avg["betas"]
            outputs["pred_smpl_params_global"]["body_pose"] = smpl_params_avg["body_pose"]

            # Finally, apply postprocess
            if do_postproc:
                # temporarily recover the original batch-dim
                outputs["pred_smpl_params_global"] = {k: v[None] for k, v in outputs["pred_smpl_params_global"].items()}
                outputs["pred_smpl_params_global"]["transl"] = pp_static_joint(outputs, self.pipeline.endecoder)
                body_pose = process_ik(outputs, self.pipeline.endecoder)
                outputs["pred_smpl_params_global"] = {k: v[0] for k, v in outputs["pred_smpl_params_global"].items()}

                outputs["pred_smpl_params_global"]["body_pose"] = body_pose[0]
                # outputs["pred_smpl_params_incam"]["body_pose"] = body_pose[0]

        if False:  # wis3d
            wis3d = make_wis3d(name="debug-rich-cap")
            smplx_model = make_smplx("rich-smplx", gender="neutral").cuda()
            gender = batch["gender"][0]
            T_w2ay = batch["T_w2ay"][0]

            # Prediction
            # add_motion_as_lines(outputs_window["pred_ayfz_motion"][bid], wis3d, name="pred_ayfz_motion")

            smplx_out = smplx_model(**pred_smpl_params_global)
            for i in range(len(smplx_out.vertices)):
                wis3d.set_scene_id(i)
                wis3d.add_mesh(smplx_out.vertices[i], smplx_model.bm.faces, name=f"pred-smplx-global")

            # GT (w)
            smplx_models = {
                "male": make_smplx("rich-smplx", gender="male").cuda(),
                "female": make_smplx("rich-smplx", gender="female").cuda(),
            }
            gt_smpl_params = {k: v[0, windows[0]] for k, v in batch["gt_smpl_params"].items()}
            gt_smplx_out = smplx_models[gender](**gt_smpl_params)

            # GT (ayfz)
            smplx_verts_ay = apply_T_on_points(gt_smplx_out.vertices, T_w2ay)
            smplx_joints_ay = apply_T_on_points(gt_smplx_out.joints, T_w2ay)
            T_ay2ayfz = compute_T_ayfz2ay(smplx_joints_ay[:1], inverse=True)[0]  # (4, 4)
            smplx_verts_ayfz = apply_T_on_points(smplx_verts_ay, T_ay2ayfz)  # (F, 22, 3)

            for i in range(len(smplx_verts_ayfz)):
                wis3d.set_scene_id(i)
                wis3d.add_mesh(smplx_verts_ayfz[i], smplx_models[gender].bm.faces, name=f"gt-smplx-ayfz")

            breakpoint()

        if False:  # o3d
            prog_keys = [
                "pred_smpl_progress",
                "pred_localjoints_progress",
                "pred_incam_localjoints_progress",
            ]
            for k in prog_keys:
                if k in outputs_window:
                    seq_out = torch.cat(
                        [v[:, :l] for v, l in zip(outputs_window[k], length)], dim=1
                    )  # (B, P, L, J, 3) -> (P, L, J, 3) -> (P, CL, J, 3)
                    outputs[k] = seq_out[None]

        return outputs

    def configure_optimizers(self):
        params = []
        for k, v in self.pipeline.named_parameters():
            if v.requires_grad:
                params.append(v)
        optimizer = self.optimizer(params=params)

        if self.scheduler_cfg["scheduler"] is None:
            return optimizer

        scheduler_cfg = dict(self.scheduler_cfg)
        scheduler_cfg["scheduler"] = instantiate(scheduler_cfg["scheduler"], optimizer=optimizer)
        return [optimizer], [scheduler_cfg]

    # ============== Utils ================= #
    def on_save_checkpoint(self, checkpoint) -> None:
        for ig_keys in self.ignored_weights_prefix:
            for k in list(checkpoint["state_dict"].keys()):
                if k.startswith(ig_keys):
                    # Log.info(f"Remove key `{ig_keys}' from checkpoint.")
                    checkpoint["state_dict"].pop(k)

    def load_pretrained_model(self, ckpt_path):
        """Load pretrained checkpoint, and assign each weight to the corresponding part."""
        Log.info(f"[PL-Trainer] Loading ckpt: {ckpt_path}")

        state_dict = torch.load(ckpt_path, "cpu")["state_dict"]
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        real_missing = []
        for k in missing:
            ignored_when_saving = any(k.startswith(ig_keys) for ig_keys in self.ignored_weights_prefix)
            if not ignored_when_saving:
                real_missing.append(k)

        if len(real_missing) > 0:
            Log.warn(f"Missing keys: {real_missing}")
        if len(unexpected) > 0:
            Log.warn(f"Unexpected keys: {unexpected}")


gvhmr_pl = builds(
    GvhmrPL,
    pipeline="${pipeline}",
    optimizer="${optimizer}",
    scheduler_cfg="${scheduler_cfg}",
    populate_full_signature=True,  # Adds all the arguments to the signature
)
MainStore.store(name="gvhmr_pl", node=gvhmr_pl, group="model/gvhmr")
