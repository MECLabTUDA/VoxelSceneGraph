import math
from functools import reduce

import torch


def affine_transformation_matrix(
        tensor_size: tuple[int, ...],
        translate: tuple[float, ...] = (0.,),
        scale: tuple[float, ...] = (1.,),
        rotate: tuple[float, ...] = (0,)
) -> torch.Tensor:
    """
    Generate an affine matrix for the specified transformation.
    The order of operation is scaling, then rotation, then translation!
    Note: the lengths of all tuple should either be one or match n_dim.
    Note: all parameters are supplied depth-first.
    :param tensor_size: size of the tensor that will be transformed (excluding batch and channels).
    :param translate: a tuple of length 2 or 3 with the amount of translation (in pixel) along each dimension.
    :param scale: a tuple of length 2 or 3 with the scaling (strictly positive) along each dimension.
    :param rotate: a tuple of length 2 or 3 with the amount of rotation (in degrees) along each dimension.
    :returns: the sampling grid with shape (1 (batch size dim), *tensor_size, n_dim).
    """
    n_dim = len(tensor_size)
    assert n_dim in [2, 3]
    # Expand 1 length tuples
    if len(translate) == 1:
        translate *= n_dim
    if len(scale) == 1:
        scale *= n_dim
    if len(rotate) == 1 and n_dim == 3:
        rotate *= n_dim
    # Assert tuple length
    assert all(length > 0 for length in tensor_size)
    assert len(translate) == n_dim
    assert len(scale) == n_dim
    assert all(s > 0 for s in scale), "Scales need to be all strictly positive"
    assert (len(rotate) == 1 and n_dim == 2) or (len(rotate) == 3 and n_dim == 3)

    # Note: VERY IMPORTANT, the order of operation needs to be scaling, then rotation, then translation!
    #       This way, all parameters for this function are with respect to the original tensor.
    # Note: This transformation matrix structure expected width-first, hence all the reversed() calls
    # Note: The content of the transformation are in normalized coordinates [-1, 1],
    #       so we need to convert them from pixel scale
    ops: list[torch.Tensor] = []

    # Transform with scale (1/s since a scale of 2 should make it twice as large)...
    scale_matrix = torch.diag(torch.tensor(tuple(1 / s for s in reversed(scale)) + (1.,))).float()
    ops.append(scale_matrix)

    # Convert angle from degrees to radian
    angles = tuple(angle / 180 * math.pi for angle in rotate)  # No need for reversed() call since we index manually
    # Rotation matrix
    # Note: since the coordinates are normalized to [-1, 1[, the center of the tensor is already at [0, 0]
    if n_dim == 2:
        rotate_matrix = torch.tensor([
            [math.cos(angles[0]), -math.sin(angles[0]), 0],
            [math.sin(angles[0]), math.cos(angles[0]), 0],
            [0, 0, 1],
        ])
        ops.append(rotate_matrix)
    else:
        # https://www.brainvoyager.com/bv/doc/UsersGuide/CoordsAndTransforms/SpatialTransformationMatrices.html
        rotate_matrix_z = torch.tensor([
            [math.cos(angles[0]), -math.sin(angles[0]), 0, 0],
            [math.sin(angles[0]), math.cos(angles[0]), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        rotate_matrix_y = torch.tensor([
            [math.cos(angles[1]), 0, -math.sin(angles[1]), 0],
            [0, 1, 0, 0],
            [math.sin(angles[1]), 0, math.cos(angles[1]), 0],
            [0, 0, 0, 1],
        ])
        rotate_matrix_x = torch.tensor([
            [1, 0, 0, 0],
            [0, math.cos(angles[2]), math.sin(angles[2]), 0],
            [0, -math.sin(angles[2]), math.cos(angles[2]), 0],
            [0, 0, 0, 1],
        ])
        ops += [rotate_matrix_z, rotate_matrix_y, rotate_matrix_x]

    # ...and translation (-... because we move in the opposite direction and *2/l to normalize)
    translate_matrix = torch.eye(n_dim + 1)
    translate_matrix[:n_dim, -1] = torch.tensor([-2 * t / l for t, l in
                                                 reversed(list(zip(translate, tensor_size)))])
    ops.append(translate_matrix)

    # Since the last row is always constant, pytorch expects the transformation matrix to be truncated
    # We also need to add a batch dim
    return reduce(lambda a, b: torch.matmul(a, b), ops)


def affine_transformation_grid(
        tensor_size: tuple[int, ...],
        translate: tuple[float, ...] = (0.,),
        scale: tuple[float, ...] = (1.,),
        rotate: tuple[float, ...] = (0,)
) -> torch.Tensor:
    """
    Generate a sampling grid for the specified affine transformation,
    which can then be used with torch.nn.functional.grid_sample.
    The order of operation is scaling, then rotation, then translation!
    Note: the lengths of all tuple should either be one or match n_dim.
    Note: all parameters are supplied depth-first.
    :param tensor_size: size of the tensor that will be transformed (excluding batch and channels).
    :param translate: a tuple of length 2 or 3 with the amount of translation (in pixel) along each dimension.
    :param scale: a tuple of length 2 or 3 with the scaling (strictly positive) along each dimension.
    :param rotate: a tuple of length 2 or 3 with the amount of rotation (in degrees) along each dimension.
    :returns: the sampling grid with shape (1 (batch size dim), *tensor_size, n_dim).
    """
    # Since the last row is always constant, pytorch expects the transformation matrix to be truncated
    # We also need to add a batch dim
    full_transform_matrix = affine_transformation_matrix(tensor_size, translate, scale, rotate)[None, :-1]
    # Add batch and dummy channel dim to the shape
    return torch.nn.functional.affine_grid(full_transform_matrix, list((1, 1) + tensor_size), align_corners=False)


def compute_bounding_box(tensor: torch.Tensor) -> torch.Tensor:
    """Compute the non-zero bounding box (z1, y1, x1, z2, y2, x2) for a ND tensor (without batch or channel dim)."""
    n_dim = tensor.dim()

    # Important to check that the tensor is not all zero for the torch.where calls.
    if not torch.any(tensor):
        # Return a bounding box of size 0 along each dim
        return torch.cat([
            torch.zeros(n_dim, dtype=torch.long, device=tensor.device),
            torch.full((n_dim,), -1, dtype=torch.long, device=tensor.device)
        ])

    non_zeros = torch.nonzero(tensor)
    return torch.stack(
        [torch.min(non_zeros[:, dim]) for dim in range(n_dim)] +
        [torch.max(non_zeros[:, dim]) for dim in range(n_dim)]
    )


def relation_matrix_to_triplets(rel_matrix: torch.LongTensor) -> torch.LongTensor:
    """
    Convert a relation matrix to #relx3 triplets.
    :param rel_matrix: #objx#obj matrix containing relation labels as integers. 0 is considered the background class.
    :return: triplets in format (subj id, obj id, pred label)
    """
    non_bg_idxs: tuple = torch.nonzero(rel_matrix, as_tuple=True)
    # noinspection PyTypeChecker
    return torch.stack(non_bg_idxs + (rel_matrix[non_bg_idxs],), dim=1)
