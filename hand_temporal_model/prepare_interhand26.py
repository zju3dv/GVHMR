"""
InterHand2.6M MANO_NeuralAnnot 전처리 파서
────────────────────────────────────────────────────────────────
입력: InterHand2.6M_{split}_MANO_NeuralAnnot.json

JSON 구조:
  {
    capture_id: {
      frame_idx: {
        "right": {"pose": [48,], "shape": [10,], "trans": [3,]} or null,
        "left":  {"pose": [48,], "shape": [10,], "trans": [3,]} or null
      }
    }
  }

출력: {split}_interhand.npz  (HO3D 파서와 동일 포맷)
  - hand_pose   : (N, 45)  float32  axis-angle, global_orient 제외
  - seq_lengths : (S,)     int32    각 시퀀스의 프레임 수
  - mean        : (45,)    float32  정규화용 평균
  - std         : (45,)    float32  정규화용 표준편차
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm


# ── 유틸 ─────────────────────────────────────────────────────────

def extract_pose(annot: dict):
    """
    단일 프레임 어노테이션에서 hand_pose(45,) 추출.
    None이면 None 반환 (해당 프레임 손 없음).
    """
    if annot is None:
        return None
    try:
        pose = np.array(annot['pose'], dtype=np.float32)  # (48,)
        return pose[3:]                                    # (45,) global_orient 제외
    except Exception:
        return None


def split_at_none(frames: list, min_len: int) -> list:
    """
    None이 끼어있는 프레임 리스트를 연속 구간으로 분할.

    예) [v, v, None, v, v, v] → [[v,v], [v,v,v]]
    None이 있는 위치에서 시퀀스를 끊어서 연속성을 보장.
    """
    sequences = []
    current   = []

    for frame in frames:
        if frame is None:
            if len(current) >= min_len:
                sequences.append(np.stack(current, axis=0))
            current = []
        else:
            current.append(frame)

    if len(current) >= min_len:
        sequences.append(np.stack(current, axis=0))

    return sequences


# ── 파서 ─────────────────────────────────────────────────────────

def parse_capture(capture_data: dict, min_len: int):
    """
    하나의 capture(시퀀스)에서 오른손·왼손 시퀀스 분할 추출.

    Returns
    -------
    right_seqs : list of np.ndarray (Ti, 45)
    left_seqs  : list of np.ndarray (Ti, 45)
    """
    # frame_idx 기준 정렬 (문자열 키 → 정수 정렬)
    sorted_frames = sorted(capture_data.items(), key=lambda x: int(x[0]))

    right_frames = [extract_pose(v.get('right')) for _, v in sorted_frames]
    left_frames  = [extract_pose(v.get('left'))  for _, v in sorted_frames]

    right_seqs = split_at_none(right_frames, min_len)
    left_seqs  = split_at_none(left_frames,  min_len)

    return right_seqs, left_seqs


def prepare_interhand(
    json_path   : str,
    output_path : str,
    min_seq_len : int  = 31,
    use_left    : bool = True,
    use_right   : bool = True,
) -> None:
    """
    InterHand2.6M 전처리 메인 함수.

    Parameters
    ----------
    json_path    : MANO_NeuralAnnot.json 경로
    output_path  : 저장할 .npz 경로
    min_seq_len  : 이 값보다 짧은 연속 구간은 제외
    use_left     : 왼손 데이터 포함 여부
    use_right    : 오른손 데이터 포함 여부
    """
    print(f"JSON 로딩 중: {json_path}")
    print("  (파일이 크면 수십 초 걸릴 수 있어요)")

    with open(json_path, 'r') as f:
        data = json.load(f)

    all_seqs      = []
    n_right_frames = 0
    n_left_frames  = 0
    n_skipped      = 0

    for capture_id, capture_data in tqdm(data.items(), desc="capture 처리"):
        right_seqs, left_seqs = parse_capture(capture_data, min_seq_len)

        if use_right:
            for seq in right_seqs:
                all_seqs.append(seq)
                n_right_frames += len(seq)

        if use_left:
            for seq in left_seqs:
                all_seqs.append(seq)
                n_left_frames += len(seq)

        # 유효 시퀀스 없는 capture 카운트
        if not right_seqs and not left_seqs:
            n_skipped += 1

    if not all_seqs:
        raise RuntimeError("유효한 시퀀스가 없습니다. json_path를 확인하세요.")

    seq_lengths = np.array([len(s) for s in all_seqs], dtype=np.int32)
    all_poses   = np.concatenate(all_seqs, axis=0).astype(np.float32)  # (N, 45)

    mean = all_poses.mean(axis=0)
    std  = all_poses.std(axis=0)
    std  = np.where(std < 1e-6, 1.0, std)

    np.savez(
        output_path,
        hand_pose   = all_poses,
        seq_lengths = seq_lengths,
        mean        = mean,
        std         = std,
    )

    print(f"\n{'='*50}")
    print(f"저장 완료 : {output_path}")
    print(f"{'='*50}")
    print(f"오른손 프레임  : {n_right_frames:>10,}")
    print(f"왼손 프레임    : {n_left_frames:>10,}")
    print(f"총 프레임      : {n_right_frames + n_left_frames:>10,}")
    print(f"총 시퀀스      : {len(all_seqs):>10,}")
    print(f"빈 capture     : {n_skipped:>10,}")
    print(f"pose 평균 범위 : [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"pose std  범위 : [{std.min():.4f},  {std.max():.4f}]")
    print(f"{'='*50}\n")


# ── 검증 ─────────────────────────────────────────────────────────

def verify_npz(npz_path: str, seq_len: int = 31) -> None:
    data        = np.load(npz_path)
    hand_pose   = data['hand_pose']
    seq_lengths = data['seq_lengths']

    n_windows = sum(max(0, L - seq_len + 1) for L in seq_lengths)

    print(f"\n[검증] {npz_path}")
    print(f"  hand_pose shape  : {hand_pose.shape}")
    print(f"  시퀀스 수         : {len(seq_lengths):,}")
    print(f"  프레임 수 합계    : {seq_lengths.sum():,}")
    print(f"  슬라이딩 윈도우   : {n_windows:,}  (seq_len={seq_len})")
    print(f"  nan 포함 여부     : {np.isnan(hand_pose).any()}")
    print(f"  inf 포함 여부     : {np.isinf(hand_pose).any()}")


# ── 진입점 ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InterHand2.6M 전처리 파서")
    parser.add_argument("--data_dir", type=str,
                        default="InterHand2.6_dataset",
                        help="MANO_NeuralAnnot.json 파일들이 있는 폴더")
    parser.add_argument("--out_dir",  type=str, default=".")
    parser.add_argument("--seq_len",  type=int, default=31)
    parser.add_argument("--no_left",  action="store_true", help="왼손 제외")
    parser.add_argument("--no_right", action="store_true", help="오른손 제외")
    args = parser.parse_args()

    use_left  = not args.no_left
    use_right = not args.no_right

    for split in ('train', 'val'):
        json_path = os.path.join(
            args.data_dir,
            f"InterHand2.6M_{split}_MANO_NeuralAnnot.json"
        )
        if not os.path.exists(json_path):
            print(f"[SKIP] 파일 없음: {json_path}")
            continue

        out_path = os.path.join(args.out_dir, f"interhand_{split}.npz")
        prepare_interhand(
            json_path   = json_path,
            output_path = out_path,
            min_seq_len = args.seq_len,
            use_left    = use_left,
            use_right   = use_right,
        )
        verify_npz(out_path, seq_len=args.seq_len)