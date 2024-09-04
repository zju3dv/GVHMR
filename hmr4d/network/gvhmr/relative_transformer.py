import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from hmr4d.configs import MainStore, builds

from hmr4d.network.base_arch.transformer.encoder_rope import EncoderRoPEBlock
from hmr4d.network.base_arch.transformer.layer import zero_module

from hmr4d.utils.net_utils import length_to_mask
from timm.models.vision_transformer import Mlp


class NetworkEncoderRoPE(nn.Module):
    def __init__(
        self,
        # x
        output_dim=151,
        max_len=120,
        # condition
        cliffcam_dim=3,
        cam_angvel_dim=6,
        imgseq_dim=1024,
        # intermediate
        latent_dim=512,
        num_layers=12,
        num_heads=8,
        mlp_ratio=4.0,
        # output
        pred_cam_dim=3,
        static_conf_dim=6,
        # training
        dropout=0.1,
        # other
        avgbeta=True,
    ):
        super().__init__()

        # input
        self.output_dim = output_dim
        self.max_len = max_len

        # condition
        self.cliffcam_dim = cliffcam_dim
        self.cam_angvel_dim = cam_angvel_dim
        self.imgseq_dim = imgseq_dim

        # intermediate
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        # ===== build model ===== #
        # Input (Kp2d)
        # Main token: map d_obs 2 to 32
        self.learned_pos_linear = nn.Linear(2, 32)
        self.learned_pos_params = nn.Parameter(torch.randn(17, 32), requires_grad=True)
        self.embed_noisyobs = Mlp(
            17 * 32, hidden_features=self.latent_dim * 2, out_features=self.latent_dim, drop=dropout
        )

        self._build_condition_embedder()

        # Transformer
        self.blocks = nn.ModuleList(
            [
                EncoderRoPEBlock(self.latent_dim, self.num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(self.num_layers)
            ]
        )

        # Output heads
        self.final_layer = Mlp(self.latent_dim, out_features=self.output_dim)
        self.pred_cam_head = pred_cam_dim > 0  # keep extra_output for easy-loading old ckpt
        if self.pred_cam_head:
            self.pred_cam_head = Mlp(self.latent_dim, out_features=pred_cam_dim)
            self.register_buffer("pred_cam_mean", torch.tensor([1.0606, -0.0027, 0.2702]), False)
            self.register_buffer("pred_cam_std", torch.tensor([0.1784, 0.0956, 0.0764]), False)

        self.static_conf_head = static_conf_dim > 0
        if self.static_conf_head:
            self.static_conf_head = Mlp(self.latent_dim, out_features=static_conf_dim)

        self.avgbeta = avgbeta

    def _build_condition_embedder(self):
        latent_dim = self.latent_dim
        dropout = self.dropout
        self.cliffcam_embedder = nn.Sequential(
            nn.Linear(self.cliffcam_dim, latent_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )
        if self.cam_angvel_dim > 0:
            self.cam_angvel_embedder = nn.Sequential(
                nn.Linear(self.cam_angvel_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim)),
            )
        if self.imgseq_dim > 0:
            self.imgseq_embedder = nn.Sequential(
                nn.LayerNorm(self.imgseq_dim),
                zero_module(nn.Linear(self.imgseq_dim, latent_dim)),
            )

    def forward(self, length, obs=None, f_cliffcam=None, f_cam_angvel=None, f_imgseq=None):
        """
        Args:
            x: None we do not use it
            timesteps: (B,)
            length: (B), valid length of x, if None then use x.shape[2]
            f_imgseq: (B, L, C)
            f_cliffcam: (B, L, 3), CLIFF-Cam parameters (bbx-detection in the full-image)
            f_noisyobs: (B, L, C), nosiy pose observation
            f_cam_angvel: (B, L, 6), Camera angular velocity
        """
        B, L, J, C = obs.shape
        assert J == 17 and C == 3

        # Main token from observation (2D pose)
        obs = obs.clone()
        visible_mask = obs[..., [2]] > 0.5  # (B, L, J, 1)
        obs[~visible_mask[..., 0]] = 0  # set low-conf to all zeros
        f_obs = self.learned_pos_linear(obs[..., :2])  # (B, L, J, 32)
        f_obs = f_obs * visible_mask + self.learned_pos_params.repeat(B, L, 1, 1) * ~visible_mask
        x = self.embed_noisyobs(f_obs.view(B, L, -1))  # (B, L, J*32) -> (B, L, C)

        # Condition
        f_to_add = []
        f_to_add.append(self.cliffcam_embedder(f_cliffcam))
        if hasattr(self, "cam_angvel_embedder"):
            f_to_add.append(self.cam_angvel_embedder(f_cam_angvel))
        if f_imgseq is not None and hasattr(self, "imgseq_embedder"):
            f_to_add.append(self.imgseq_embedder(f_imgseq))

        for f_delta in f_to_add:
            x = x + f_delta

        # Setup length and make padding mask
        assert B == length.size(0)
        pmask = ~length_to_mask(length, L)  # (B, L)

        if L > self.max_len:
            attnmask = torch.ones((L, L), device=x.device, dtype=torch.bool)
            for i in range(L):
                min_ind = max(0, i - self.max_len // 2)
                max_ind = min(L, i + self.max_len // 2)
                max_ind = max(self.max_len, max_ind)
                min_ind = min(L - self.max_len, min_ind)
                attnmask[i, min_ind:max_ind] = False
        else:
            attnmask = None

        # Transformer
        for block in self.blocks:
            x = block(x, attn_mask=attnmask, tgt_key_padding_mask=pmask)

        # Output
        sample = self.final_layer(x)  # (B, L, C)
        if self.avgbeta:
            betas = (sample[..., 126:136] * (~pmask[..., None])).sum(1) / length[:, None]  # (B, C)
            betas = repeat(betas, "b c -> b l c", l=L)
            sample = torch.cat([sample[..., :126], betas, sample[..., 136:]], dim=-1)

        # Output (extra)
        pred_cam = None
        if self.pred_cam_head:
            pred_cam = self.pred_cam_head(x)
            pred_cam = pred_cam * self.pred_cam_std + self.pred_cam_mean
            torch.clamp_min_(pred_cam[..., 0], 0.25)  # min_clamp s to 0.25 (prevent negative prediction)

        static_conf_logits = None
        if self.static_conf_head:
            static_conf_logits = self.static_conf_head(x)  # (B, L, C')

        output = {
            "pred_context": x,
            "pred_x": sample,
            "pred_cam": pred_cam,
            "static_conf_logits": static_conf_logits,
        }
        return output


# Add to MainStore
group_name = "network/gvhmr"
MainStore.store(
    name="relative_transformer",
    node=builds(NetworkEncoderRoPE, populate_full_signature=True),
    group=group_name,
)
