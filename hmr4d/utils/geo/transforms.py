import torch


def axis_rotate_to_matrix(angle, axis="x"):
    """Get rotation matrix for rotating around one axis
    Args:
        angle: (N, 1)
    Returns:
        R: (N, 3, 3)
    """
    if isinstance(angle, float):
        angle = torch.tensor([angle], dtype=torch.float)

    c = torch.cos(angle)
    s = torch.sin(angle)
    z = torch.zeros_like(angle)
    o = torch.ones_like(angle)
    if axis == "x":
        R = torch.stack([o, z, z, z, c, -s, z, s, c], dim=1).view(-1, 3, 3)
    elif axis == "y":
        R = torch.stack([c, z, s, z, o, z, -s, z, c], dim=1).view(-1, 3, 3)
    else:
        assert axis == "z"
        R = torch.stack([c, -s, z, s, c, z, z, z, o], dim=1).view(-1, 3, 3)
    return R
