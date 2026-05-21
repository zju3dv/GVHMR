from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation


SMPL_JOINTS = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]


DEFAULT_ALIASES = {
    "pelvis": ["hips", "hip", "pelvis", "root", "base"],
    "left_hip": ["leftupleg", "leftthigh", "lefthip", "lhip", "lthigh"],
    "right_hip": ["rightupleg", "rightthigh", "righthip", "rhip", "rthigh"],
    "spine1": ["spine", "spine1", "lowerback", "abdomen"],
    "left_knee": ["leftleg", "leftknee", "leftshin", "leftcalf", "lleg", "lknee"],
    "right_knee": ["rightleg", "rightknee", "rightshin", "rightcalf", "rleg", "rknee"],
    "spine2": ["spine1", "spine2", "chest", "upperback", "thorax"],
    "left_ankle": ["leftfoot", "leftankle", "lfoot", "lankle"],
    "right_ankle": ["rightfoot", "rightankle", "rfoot", "rankle"],
    "spine3": ["spine2", "spine3", "upperchest", "chest2"],
    "left_foot": ["lefttoebase", "lefttoe", "lefttoes", "ltoebase", "ltoe"],
    "right_foot": ["righttoebase", "righttoe", "righttoes", "rtoebase", "rtoe"],
    "neck": ["neck", "neck1"],
    "left_collar": ["leftshoulder", "leftclavicle", "lshoulder", "lclavicle"],
    "right_collar": ["rightshoulder", "rightclavicle", "rshoulder", "rclavicle"],
    "head": ["head", "headtop", "headend"],
    "left_shoulder": ["leftarm", "leftupperarm", "larm", "lupperarm"],
    "right_shoulder": ["rightarm", "rightupperarm", "rarm", "rupperarm"],
    "left_elbow": ["leftforearm", "leftlowerarm", "leftelbow", "lforearm", "lelbow"],
    "right_elbow": ["rightforearm", "rightlowerarm", "rightelbow", "rforearm", "relbow"],
    "left_wrist": ["lefthand", "leftwrist", "lhand", "lwrist"],
    "right_wrist": ["righthand", "rightwrist", "rhand", "rwrist"],
}


AXIS_TO_VECTOR = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}


@dataclass
class BvhJoint:
    name: str
    parent: int | None
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    channels: list[str] = field(default_factory=list)
    channel_start: int = 0
    children: list[int] = field(default_factory=list)


@dataclass
class BvhMotion:
    joints: list[BvhJoint]
    frames: int
    frame_time: float
    values: np.ndarray
    channel_count: int


def normalize_name(name: str) -> str:
    name = name.split(":")[-1]
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def load_aliases(path: Path | None) -> dict[str, list[str]]:
    aliases = {k: list(v) for k, v in DEFAULT_ALIASES.items()}
    if path is None:
        return aliases

    with path.open("r", encoding="utf-8") as handle:
        user_aliases = json.load(handle)
    for key, values in user_aliases.items():
        if key not in aliases:
            raise ValueError(f"Unknown SMPL joint in alias file: {key}")
        if isinstance(values, str):
            values = [values]
        aliases[key] = list(values) + aliases[key]
    return aliases


def tokenize_hierarchy(text: str) -> list[str]:
    return re.findall(r"[{}]|[^\s{}]+", text)


def skip_block(tokens: list[str], pos: int) -> int:
    if tokens[pos] != "{":
        raise ValueError("Expected '{' while skipping BVH block.")
    depth = 1
    pos += 1
    while pos < len(tokens) and depth > 0:
        if tokens[pos] == "{":
            depth += 1
        elif tokens[pos] == "}":
            depth -= 1
        pos += 1
    if depth != 0:
        raise ValueError("Unclosed BVH block.")
    return pos


def parse_bvh(path: Path) -> BvhMotion:
    text = path.read_text(encoding="utf-8", errors="ignore")
    motion_match = re.search(r"\bMOTION\b", text)
    if motion_match is None:
        raise ValueError(f"BVH MOTION section not found: {path}")

    hierarchy_text = text[: motion_match.start()]
    motion_text = text[motion_match.end() :]
    tokens = tokenize_hierarchy(hierarchy_text)
    if not tokens or tokens[0] != "HIERARCHY":
        raise ValueError(f"BVH HIERARCHY section not found: {path}")

    joints: list[BvhJoint] = []
    channel_count = 0

    def parse_joint(pos: int, parent: int | None) -> int:
        nonlocal channel_count
        kind = tokens[pos]
        if kind not in {"ROOT", "JOINT"}:
            raise ValueError(f"Expected ROOT or JOINT, got {kind}")
        name = tokens[pos + 1]
        pos += 2
        if tokens[pos] != "{":
            raise ValueError(f"Expected '{{' after {kind} {name}")
        pos += 1

        joint_idx = len(joints)
        joints.append(BvhJoint(name=name, parent=parent))
        if parent is not None:
            joints[parent].children.append(joint_idx)

        while pos < len(tokens):
            tok = tokens[pos]
            if tok == "}":
                return pos + 1
            if tok == "OFFSET":
                joints[joint_idx].offset = np.array(
                    [float(tokens[pos + 1]), float(tokens[pos + 2]), float(tokens[pos + 3])],
                    dtype=np.float64,
                )
                pos += 4
                continue
            if tok == "CHANNELS":
                n_channels = int(tokens[pos + 1])
                channels = tokens[pos + 2 : pos + 2 + n_channels]
                joints[joint_idx].channels = channels
                joints[joint_idx].channel_start = channel_count
                channel_count += n_channels
                pos += 2 + n_channels
                continue
            if tok == "JOINT":
                pos = parse_joint(pos, joint_idx)
                continue
            if tok == "End" and pos + 1 < len(tokens) and tokens[pos + 1] == "Site":
                pos += 2
                pos = skip_block(tokens, pos)
                continue
            raise ValueError(f"Unsupported BVH token near {path}: {tok}")

        raise ValueError(f"Unclosed joint block in {path}")

    pos = 1
    pos = parse_joint(pos, None)
    if pos != len(tokens):
        leftovers = " ".join(tokens[pos : pos + 8])
        raise ValueError(f"Unexpected tokens after hierarchy: {leftovers}")

    frames_match = re.search(r"Frames:\s*(\d+)", motion_text)
    frame_time_match = re.search(r"Frame\s+Time:\s*([0-9eE+\-.]+)", motion_text)
    if frames_match is None or frame_time_match is None:
        raise ValueError(f"BVH frame metadata not found: {path}")

    frames = int(frames_match.group(1))
    frame_time = float(frame_time_match.group(1))
    motion_values_text = motion_text[frame_time_match.end() :]
    flat_values = np.fromstring(motion_values_text, sep=" ", dtype=np.float64)
    expected = frames * channel_count
    if flat_values.size != expected:
        raise ValueError(
            f"BVH motion value count mismatch for {path}: got {flat_values.size}, expected {expected} "
            f"({frames} frames * {channel_count} channels)"
        )

    return BvhMotion(
        joints=joints,
        frames=frames,
        frame_time=frame_time,
        values=flat_values.reshape(frames, channel_count),
        channel_count=channel_count,
    )


def rotation_between_vectors(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = src / np.linalg.norm(src)
    dst = dst / np.linalg.norm(dst)
    cross = np.cross(src, dst)
    dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    if np.isclose(dot, 1.0):
        return np.eye(3, dtype=np.float64)
    if np.isclose(dot, -1.0):
        axis = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(src, axis))) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = np.cross(src, axis)
        axis = axis / np.linalg.norm(axis)
        return Rotation.from_rotvec(axis * np.pi).as_matrix()
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - dot) / (np.linalg.norm(cross) ** 2))


def get_basis_conversion(source_up: str, target_up: str) -> np.ndarray:
    source_up = source_up.lower()
    target_up = target_up.lower()
    if source_up not in AXIS_TO_VECTOR or target_up not in AXIS_TO_VECTOR:
        raise ValueError("--source-up and --target-up must be one of: x, y, z")
    return rotation_between_vectors(AXIS_TO_VECTOR[source_up], AXIS_TO_VECTOR[target_up])


def channel_matrix(motion: BvhMotion, joint_idx: int, euler_mode: str) -> np.ndarray:
    joint = motion.joints[joint_idx]
    if not joint.channels:
        return np.repeat(np.eye(3, dtype=np.float64)[None], motion.frames, axis=0)

    rotation_axes = []
    rotation_values = []
    for channel_offset, channel in enumerate(joint.channels):
        lower = channel.lower()
        if lower.endswith("rotation"):
            rotation_axes.append(lower[0])
            rotation_values.append(motion.values[:, joint.channel_start + channel_offset])

    if not rotation_axes:
        return np.repeat(np.eye(3, dtype=np.float64)[None], motion.frames, axis=0)

    axis_order = "".join(rotation_axes)
    if euler_mode == "intrinsic":
        axis_order = axis_order.upper()
    elif euler_mode != "extrinsic":
        raise ValueError("--euler-mode must be intrinsic or extrinsic")

    angles = np.stack(rotation_values, axis=1)
    return Rotation.from_euler(axis_order, angles, degrees=True).as_matrix()


def channel_translation(motion: BvhMotion, joint_idx: int) -> np.ndarray:
    joint = motion.joints[joint_idx]
    out = np.zeros((motion.frames, 3), dtype=np.float64)
    for channel_offset, channel in enumerate(joint.channels):
        lower = channel.lower()
        if lower == "xposition":
            out[:, 0] = motion.values[:, joint.channel_start + channel_offset]
        elif lower == "yposition":
            out[:, 1] = motion.values[:, joint.channel_start + channel_offset]
        elif lower == "zposition":
            out[:, 2] = motion.values[:, joint.channel_start + channel_offset]
    return out


def find_joint_indices(motion: BvhMotion, aliases: dict[str, list[str]]) -> tuple[dict[str, int | None], list[str]]:
    normalized = [(normalize_name(joint.name), idx) for idx, joint in enumerate(motion.joints)]
    mapping: dict[str, int | None] = {}
    warnings: list[str] = []

    for smpl_name in SMPL_JOINTS:
        candidates = [normalize_name(v) for v in aliases[smpl_name]]
        matched: int | None = None

        for alias in candidates:
            exact = [idx for norm, idx in normalized if norm == alias]
            if exact:
                matched = exact[0]
                break

        if matched is None:
            for alias in candidates:
                suffix = [idx for norm, idx in normalized if norm.endswith(alias)]
                if suffix:
                    matched = suffix[0]
                    break

        if matched is None:
            warnings.append(f"missing BVH joint for SMPL joint '{smpl_name}', using zero rotation")
        mapping[smpl_name] = matched

    reverse: dict[int, list[str]] = {}
    for smpl_name, joint_idx in mapping.items():
        if joint_idx is not None:
            reverse.setdefault(joint_idx, []).append(smpl_name)
    for joint_idx, smpl_names in reverse.items():
        if len(smpl_names) > 1:
            warnings.append(
                f"BVH joint '{motion.joints[joint_idx].name}' is mapped to multiple SMPL joints: "
                + ", ".join(smpl_names)
            )

    return mapping, warnings


def matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(matrix).as_rotvec().astype(np.float32)


def convert_bvh_to_smpl_record(
    path: Path,
    aliases: dict[str, list[str]],
    root_scale: float,
    source_up: str,
    target_up: str,
    euler_mode: str,
    rotate_all_joints: bool,
    strict: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    motion = parse_bvh(path)
    mapping, warnings = find_joint_indices(motion, aliases)
    if strict and warnings:
        raise ValueError("Strict BVH mapping failed:\n" + "\n".join(warnings))

    basis = get_basis_conversion(source_up, target_up)
    pose = np.zeros((motion.frames, 66), dtype=np.float32)

    root_idx = mapping["pelvis"]
    if root_idx is None:
        root_idx = 0
    trans = channel_translation(motion, root_idx) * root_scale
    trans = (basis @ trans.T).T.astype(np.float32)

    for smpl_idx, smpl_name in enumerate(SMPL_JOINTS):
        bvh_idx = mapping[smpl_name]
        if bvh_idx is None:
            continue

        rotmat = channel_matrix(motion, bvh_idx, euler_mode)
        if rotate_all_joints or smpl_idx == 0:
            rotmat = basis[None] @ rotmat @ basis.T[None]
        rotvec = matrix_to_rotvec(rotmat)

        if smpl_idx == 0:
            pose[:, :3] = rotvec
        else:
            start = 3 + (smpl_idx - 1) * 3
            pose[:, start : start + 3] = rotvec

    beta = np.zeros(10, dtype=np.float32)
    record = {
        "pose": torch.from_numpy(pose),
        "trans": torch.from_numpy(trans),
        "beta": torch.from_numpy(beta),
    }
    meta = {
        "source_path": str(path),
        "frames": motion.frames,
        "fps": 1.0 / motion.frame_time if motion.frame_time > 0 else None,
        "frame_time": motion.frame_time,
        "root_scale": root_scale,
        "source_up": source_up,
        "target_up": target_up,
        "euler_mode": euler_mode,
        "rotate_all_joints": rotate_all_joints,
        "joint_map": {
            smpl_name: (motion.joints[joint_idx].name if joint_idx is not None else None)
            for smpl_name, joint_idx in mapping.items()
        },
        "warnings": warnings,
    }
    return record, meta


def collect_bvh_files(inputs: list[str], recursive: bool) -> list[Path]:
    files: list[Path] = []
    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if path.is_file():
            if path.suffix.lower() != ".bvh":
                raise ValueError(f"Input file is not a .bvh file: {path}")
            files.append(path)
        elif path.is_dir():
            pattern = "**/*.bvh" if recursive else "*.bvh"
            files.extend(sorted(path.glob(pattern)))
        else:
            raise FileNotFoundError(path)
    return sorted(dict.fromkeys(files))


def sequence_key(path: Path, prefix: str, existing: set[str]) -> str:
    base = normalize_name(path.stem) or "sequence"
    key = f"{prefix}/{base}" if prefix else base
    if key not in existing:
        return key
    idx = 2
    while f"{key}_{idx}" in existing:
        idx += 1
    return f"{key}_{idx}"


def load_existing_output(path: Path, merge_existing: bool) -> dict[str, Any]:
    if not path.exists():
        return {}
    if not merge_existing:
        raise FileExistsError(f"Output already exists. Use --merge-existing to append/overwrite safely: {path}")
    loaded = torch.load(path, map_location="cpu")
    if not isinstance(loaded, dict):
        raise ValueError(f"Existing output is not a dict: {path}")
    return loaded


def save_report(report_path: Path | None, report: dict[str, Any]) -> None:
    if report_path is None:
        return
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert BVH motion files to the AMASS-like smplxpose_v2.pth format used by "
            "hmr4d.dataset.pure_motion.amass.AmassDataset."
        )
    )
    parser.add_argument("inputs", nargs="+", help="BVH file(s) or directory path(s).")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="inputs/AMASS/hmr4d_support/smplxpose_v2.pth",
        help=(
            "Output .pth path. The default is the path read by the existing "
            "hmr4d.dataset.pure_motion.amass.AmassDataset loader."
        ),
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively search .bvh files in input directories.")
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Merge into an existing output .pth instead of failing when the file already exists.",
    )
    parser.add_argument("--key-prefix", type=str, default="bvh", help="Prefix for sequence keys stored in the .pth file.")
    parser.add_argument(
        "--root-scale",
        type=float,
        default=0.01,
        help="Scale applied to BVH root translation. Use 0.01 for centimeter BVH files, 1.0 for meter BVH files.",
    )
    parser.add_argument(
        "--source-up",
        choices=("x", "y", "z"),
        default="y",
        help="Up axis in the BVH file. Most BVH/Mixamo files are y-up.",
    )
    parser.add_argument(
        "--target-up",
        choices=("x", "y", "z"),
        default="z",
        help=(
            "Up axis to write. Keep the default z-up for the existing AmassDataset, because it later converts "
            "AMASS z-up to GVHMR y-up internally."
        ),
    )
    parser.add_argument(
        "--euler-mode",
        choices=("intrinsic", "extrinsic"),
        default="intrinsic",
        help="Euler interpretation for BVH rotation channels.",
    )
    parser.add_argument(
        "--no-rotate-all-joints",
        action="store_true",
        help="Only convert root orientation to target-up. By default all mapped local joint rotations are basis-converted.",
    )
    parser.add_argument(
        "--alias-json",
        type=str,
        default=None,
        help=(
            "Optional JSON mapping from SMPL joint names to extra BVH aliases. "
            "Example: {\"left_hip\": [\"mixamorig:LeftUpLeg\"]}"
        ),
    )
    parser.add_argument("--strict", action="store_true", help="Fail if any SMPL body joint cannot be mapped.")
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Optional JSON report path containing per-file joint mappings and warnings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bvh_files = collect_bvh_files(args.inputs, args.recursive)
    if not bvh_files:
        raise FileNotFoundError("No .bvh files found.")

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    aliases = load_aliases(Path(args.alias_json).expanduser().resolve() if args.alias_json else None)
    out_dict = load_existing_output(output, args.merge_existing)
    report: dict[str, Any] = {"output": str(output), "files": {}}

    existing_keys = set(out_dict.keys())
    for bvh_path in bvh_files:
        record, meta = convert_bvh_to_smpl_record(
            path=bvh_path,
            aliases=aliases,
            root_scale=args.root_scale,
            source_up=args.source_up,
            target_up=args.target_up,
            euler_mode=args.euler_mode,
            rotate_all_joints=not args.no_rotate_all_joints,
            strict=args.strict,
        )
        key = sequence_key(bvh_path, args.key_prefix, existing_keys)
        existing_keys.add(key)
        out_dict[key] = record
        report["files"][key] = meta

        warning_count = len(meta["warnings"])
        suffix = f", warnings={warning_count}" if warning_count else ""
        print(f"[BVH->SMPL] {bvh_path} -> {key}: frames={record['pose'].shape[0]}{suffix}")

    torch.save(out_dict, output)
    save_report(Path(args.report).expanduser().resolve() if args.report else None, report)
    print(f"[BVH->SMPL] Saved {len(out_dict)} sequence(s) to {output}")


if __name__ == "__main__":
    main()
