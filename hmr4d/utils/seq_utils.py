import torch
import numpy as np

# def get_frame_id_list_from_mask(mask):
#     """
#     Args:
#         mask (F,), bool.
#     Return:
#         frame_id_list: List of frame_ids.
#     """
#     frame_id_list = []
#     i = 0
#     while i < len(mask):
#         if not mask[i]:
#             i += 1
#         else:
#             j = i
#             while j < len(mask) and mask[j]:
#                 j += 1
#             frame_id_list.append(torch.arange(i, j))
#             i = j

#     return frame_id_list


# From GPT
def get_frame_id_list_from_mask(mask):
    # batch=64, 0.13s
    """
    Vectorized approach to get frame id list from a boolean mask.

    Args:
        mask (F,), bool tensor: Mask array where `True` indicates a frame to be processed.

    Returns:
        frame_id_list: List of torch.Tensors, each tensor containing continuous indices where mask is True.
    """
    # Find the indices where the mask changes from False to True and vice versa
    padded_mask = torch.cat(
        [torch.tensor([False], device=mask.device), mask, torch.tensor([False], device=mask.device)]
    )
    diffs = torch.diff(padded_mask.int())
    starts = (diffs == 1).nonzero(as_tuple=False).squeeze()
    ends = (diffs == -1).nonzero(as_tuple=False).squeeze()
    if starts.numel() == 0:
        return []
    if starts.numel() == 1:
        starts = starts.reshape(-1)
        ends = ends.reshape(-1)

    # Create list of ranges
    frame_id_list = [torch.arange(start, end) for start, end in zip(starts, ends)]
    return frame_id_list


def get_batch_frame_id_lists_from_mask_BLC(masks):
    # batch=64, 0.10s
    """
    处理三维掩码数组，为每个批次和通道提取连续True区段的索引列表。

    参数:
        masks (B, L, C), 布尔张量：每个元素代表一个掩码，True表示需要处理的帧。

    返回:
        batch_frame_id_lists: 对应于每个批次和每个通道的帧id列表的嵌套列表。
    """
    B, L, C = masks.size()
    # 在序列长度两端添加一个False
    padded_masks = torch.cat(
        [
            torch.zeros((B, 1, C), dtype=torch.bool, device=masks.device),
            masks,
            torch.zeros((B, 1, C), dtype=torch.bool, device=masks.device),
        ],
        dim=1,
    )
    # 计算差分来找到True区段的起始和结束点
    diffs = torch.diff(padded_masks.int(), dim=1)
    starts = (diffs == 1).nonzero(as_tuple=True)
    ends = (diffs == -1).nonzero(as_tuple=True)

    # 初始化返回列表
    batch_frame_id_lists = [[[] for _ in range(C)] for _ in range(B)]
    for b in range(B):
        for c in range(C):
            batch_start = starts[0][(starts[0] == b) & (starts[2] == c)]
            batch_end = ends[0][(ends[0] == b) & (ends[2] == c)]
            # 确保start和end都是1维张量
            batch_frame_id_lists[b][c] = [
                torch.arange(start.item(), end.item()) for start, end in zip(batch_start, batch_end)
            ]

    return batch_frame_id_lists


def get_frame_id_list_from_frame_id(frame_id):
    mask = torch.zeros(frame_id[-1] + 1, dtype=torch.bool)
    mask[frame_id] = True
    frame_id_list = get_frame_id_list_from_mask(mask)
    return frame_id_list


def rearrange_by_mask(x, mask):
    """
    x (L, *)
    mask (M,), M >= L
    """
    M = mask.size(0)
    L = x.size(0)
    if M == L:
        return x
    assert M > L
    assert mask.sum() == L
    x_rearranged = torch.zeros((M, *x.size()[1:]), dtype=x.dtype, device=x.device)
    x_rearranged[mask] = x
    return x_rearranged


def frame_id_to_mask(frame_id, max_len):
    mask = torch.zeros(max_len, dtype=torch.bool)
    mask[frame_id] = True
    return mask


def mask_to_frame_id(mask):
    frame_id = torch.where(mask)[0]
    return frame_id


def linear_interpolate_frame_ids(data, frame_id_list):
    data = data.clone()
    for i, invalid_frame_ids in enumerate(frame_id_list):
        # interplate between prev, next
        # if at beginning or end, use the same value
        if invalid_frame_ids[0] - 1 < 0 or invalid_frame_ids[-1] + 1 >= len(data):
            if invalid_frame_ids[0] - 1 < 0:
                data[invalid_frame_ids] = data[invalid_frame_ids[-1] + 1].clone()
            else:
                data[invalid_frame_ids] = data[invalid_frame_ids[0] - 1].clone()
        else:
            prev = data[invalid_frame_ids[0] - 1]
            next = data[invalid_frame_ids[-1] + 1]
            data[invalid_frame_ids] = (
                torch.linspace(0, 1, len(invalid_frame_ids) + 2)[1:-1][:, None] * (next - prev)[None] + prev[None]
            )
    return data


def linear_interpolate(data, N_middle_frames):
    """
    Args:
        data: (2, C)
    Returns:
        data_interpolated: (1+N+1, C)
    """
    prev = data[0]
    next = data[1]
    middle = torch.linspace(0, 1, N_middle_frames + 2)[1:-1][:, None] * (next - prev)[None] + prev[None]  # (N, C)
    data_interpolated = torch.cat([data[0][None], middle, data[1][None]], dim=0)  # (1+N+1, C)
    return data_interpolated


def find_top_k_span(mask, k=3):
    """
    Args:
        mask: (L,)
    Return:
        topk_span: List of tuple, usage: [start, end)
    """
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    if mask.sum() == 0:
        return []
    mask = mask.clone().float()
    mask = torch.cat([mask.new([0]), mask, mask.new([0])])
    diff = mask[1:] - mask[:-1]
    start = torch.where(diff == 1)[0]
    end = torch.where(diff == -1)[0]
    assert len(start) == len(end)
    span_lengths = end - start
    span_lengths, idx = span_lengths.sort(descending=True)
    start = start[idx]
    end = end[idx]
    return list(zip(start.tolist(), end.tolist()))[:k]
