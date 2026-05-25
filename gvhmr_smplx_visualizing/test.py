import torch

PT_PATH = "hmr4d_results.pt"


def summarize_value(v, indent=0, max_list_preview=3):
    prefix = " " * indent

    if isinstance(v, torch.Tensor):
        print(f"{prefix}Tensor shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device}")
    elif isinstance(v, dict):
        print(f"{prefix}dict with {len(v)} keys")
        for k, subv in v.items():
            print(f"{prefix}- key: {k}")
            summarize_value(subv, indent + 4, max_list_preview)
    elif isinstance(v, list):
        print(f"{prefix}list len={len(v)}")
        for i, item in enumerate(v[:max_list_preview]):
            print(f"{prefix}  [{i}]")
            summarize_value(item, indent + 4, max_list_preview)
        if len(v) > max_list_preview:
            print(f"{prefix}  ... ({len(v) - max_list_preview} more items)")
    elif isinstance(v, tuple):
        print(f"{prefix}tuple len={len(v)}")
        for i, item in enumerate(v[:max_list_preview]):
            print(f"{prefix}  ({i})")
            summarize_value(item, indent + 4, max_list_preview)
        if len(v) > max_list_preview:
            print(f"{prefix}  ... ({len(v) - max_list_preview} more items)")
    else:
        print(f"{prefix}{type(v).__name__}: {v}")


def main():
    data = torch.load(PT_PATH, map_location="cpu")

    print("=" * 80)
    print("Loaded type:", type(data))
    print("=" * 80)

    summarize_value(data, indent=0)


if __name__ == "__main__":
    main()