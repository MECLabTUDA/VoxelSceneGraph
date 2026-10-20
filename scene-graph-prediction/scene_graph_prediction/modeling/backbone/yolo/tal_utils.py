# Ultralytics YOLO, AGPL-3.0 license
import numpy as np
import torch


def make_anchors(
        n_dim: int,
        features: list[torch.Tensor] | torch.Tensor,
        strides: list[int] | torch.Tensor,
        grid_cell_offset: float = 0.5
):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert features is not None
    dtype, device = features[0].dtype, features[0].device
    for i, stride in enumerate(strides):
        dhw = features[i].shape[n_dim:] \
            if isinstance(features, list) \
            else tuple(int(features[i][d]) for d in range(n_dim))
        s_zyx = [torch.arange(end=l, device=device, dtype=dtype) + grid_cell_offset for l in dhw]
        s_zyx = torch.meshgrid(*s_zyx, indexing="ij")
        anchor_points.append(torch.stack(s_zyx, -1).view(-1, 2))
        stride_tensor.append(torch.full((np.prod(dhw), 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, zyxdhw=True, dim=-1):
    """Transform distance(ltrb) to box(zyxdhw or zyxzyx)."""
    lt, rb = distance.chunk(2, dim)
    z1y1x1 = anchor_points - lt
    z2y2x2 = anchor_points + rb
    if zyxdhw:
        center_zyx = (z1y1x1 + z2y2x2) / 2
        wh = z2y2x2 - z1y1x1
        return torch.cat((center_zyx, wh), dim)  # zyxdhw bbox
    return torch.cat((z1y1x1, z2y2x2), dim)  # zyxzyx bbox


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(zyxzyx) to dist(ltrb)."""
    z1y1x1, z2y2x2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - z1y1x1, z2y2x2 - anchor_points), -1).clamp_(0, reg_max - 0.01)  # dist (lt, rb)
