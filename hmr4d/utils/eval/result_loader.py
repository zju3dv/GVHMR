from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def torch_load_file(path: str | Path, map_location: str = "cpu") -> Any:
    path = Path(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def infer_num_frames_from_smpl_params(smpl_params: dict[str, Any]) -> int:
    for value in smpl_params.values():
        if torch.is_tensor(value) and value.ndim > 0:
            return int(value.shape[0])
    raise ValueError("Could not infer num_frames from SMPL params.")


def select_smpl_params(result: dict[str, Any], space: str = "incam") -> dict[str, Any]:
    if space not in {"incam", "global"}:
        raise ValueError(f"Unknown space: {space}")
    key = f"smpl_params_{space}"
    if key not in result:
        raise KeyError(f"Missing key in result: {key}")
    return result[key]


def load_result(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    result = torch_load_file(path, map_location=map_location)
    if not isinstance(result, dict):
        raise TypeError(f"Expected dict result, got: {type(result)}")

    has_incam = "smpl_params_incam" in result
    has_global = "smpl_params_global" in result
    if not has_incam and not has_global:
        raise KeyError("Result does not contain smpl_params_incam or smpl_params_global.")

    base_params = result.get("smpl_params_incam") or result.get("smpl_params_global")
    num_frames = infer_num_frames_from_smpl_params(base_params)
    result_type = "merged" if "cap_merge_meta" in result else "gvhmr"

    return {
        "path": str(Path(path).resolve()),
        "result_type": result_type,
        "num_frames": num_frames,
        "smpl_params_incam": result.get("smpl_params_incam"),
        "smpl_params_global": result.get("smpl_params_global"),
        "K_fullimg": result.get("K_fullimg"),
        "cap_merge_meta": result.get("cap_merge_meta"),
        "raw": result,
    }
