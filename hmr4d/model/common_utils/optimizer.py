from torch.optim import AdamW, Adam
from hmr4d.configs import MainStore, builds


optimizer_cfgs = {
    "adam_1e-3": builds(Adam, lr=1e-3, zen_partial=True),
    "adam_2e-4": builds(Adam, lr=2e-4, zen_partial=True),
    "adamw_2e-4": builds(AdamW, lr=2e-4, zen_partial=True),
    "adamw_1e-4": builds(AdamW, lr=1e-4, zen_partial=True),
    "adamw_5e-5": builds(AdamW, lr=5e-5, zen_partial=True),
    "adamw_1e-5": builds(AdamW, lr=1e-5, zen_partial=True),
    # zero-shot text-to-image generation
    "adamw_1e-3_dalle": builds(AdamW, lr=1e-3, weight_decay=1e-4, zen_partial=True),
}

for name, cfg in optimizer_cfgs.items():
    MainStore.store(name=name, node=cfg, group=f"optimizer")
