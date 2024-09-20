from typing import TypedDict

# Types as defined in these definitions

# (expansion, c, nb of times this is repeated, stride (ignored for the first block))
BlockCfg = tuple[int, int, int, int]

StageCfg = list[BlockCfg]


# Note FBNet supports a keypoint head, the stage(s) used for the kpts head still needs to be determined...


# noinspection DuplicatedCode
class ArchStages(TypedDict):
    first: list[int]  # Length to or 3: [c, s, kernel (Optional)]
    stages: list[StageCfg]  # [(t, c, n, s)] for each stage
    last: tuple[int, float]  # [c, channel_scale]
    backbone: list[int]
    rpn: list[int]
    bbox: list[int]  # Needs to match ROIHeadName.to_arch_head_name
    mask: list[int]  # Needs to match ROIHeadName.to_arch_head_name
    kpts: list[int]  # Needs to match ROIHeadName.to_arch_head_name


class ModelArch(TypedDict):
    """
    Each element of stage_op_types should have the same length
    as their corresponding element in arch_stages/stages.
    """
    stage_op_types: list[list[str]]  # names for each stage
    arch_stages: ArchStages


# Types as expanded in the builder
class ExpandedBlockCfg(TypedDict):
    stage_idx: int
    block_idx: int
    block: BlockCfg  # Stages have been expanded


class ExpandedBlockCfgWithStageOp(TypedDict):
    stage_idx: int
    block_idx: int
    block: BlockCfg  # Stages have been expanded
    stage_op_type: str


# noinspection DuplicatedCode
class UnifiedArch(TypedDict):
    first: list[int]
    stages: list[ExpandedBlockCfgWithStageOp]
    last: tuple[int, float]
    backbone: list[int]
    rpn: list[int]
    bbox: list[int]  # Needs to match ROIHeadName.to_arch_head_name
    mask: list[int]  # Needs to match ROIHeadName.to_arch_head_name
    kpts: list[int]  # Needs to match ROIHeadName.to_arch_head_name


MODEL_ARCH: dict[str, ModelArch] = {
    "default": {
        "stage_op_types": [
            # stage 0
            ["ir_k3"],
            # stage 1
            ["ir_k3"] * 2,
            # stage 2
            ["ir_k3"] * 3,
            # stage 3
            ["ir_k3"] * 7,
            # stage 4, bbox head
            ["ir_k3"] * 4,
            # stage 5, rpn
            ["ir_k3"] * 3,
            # stage 5, mask head
            ["ir_k3"] * 5,
        ],
        "arch_stages": {
            "first": [32, 2],
            "stages": [
                # [(t, c, n, s)]
                # stage 0
                [(1, 16, 1, 1)],
                # stage 1
                [(6, 24, 2, 2)],
                # stage 2
                [(6, 32, 3, 2)],
                # stage 3
                [(6, 64, 4, 2), (6, 96, 3, 1)],
                # stage 4, bbox head
                [(4, 160, 1, 2), (6, 160, 2, 1), (6, 240, 1, 1)],
                # [(6, 160, 3, 2], [6, 320, 1, 1]],
                # stage 5, rpn head
                [(6, 96, 3, 1)],
                # stage 6, mask head
                [(4, 160, 1, 1), (6, 160, 3, 1), (3, 80, 1, -2)],
            ],
            # [c, channel_scale]
            "last": (0, 0.0),
            "backbone": [0, 1, 2, 3],
            "rpn": [5],
            "bbox": [4],
            "mask": [6],
            "kpts": []
        },
    },
    "xirb16d_dsmask": {
        "stage_op_types": [
            # stage 0
            ["ir_k3"],
            # stage 1
            ["ir_k3"] * 2,
            # stage 2
            ["ir_k3"] * 3,
            # stage 3
            ["ir_k3"] * 7,
            # stage 4, bbox head
            ["ir_k3"] * 4,
            # stage 5, mask head
            ["ir_k3"] * 5,
            # stage 6, rpn
            ["ir_k3"] * 3,
        ],
        "arch_stages": {
            "first": [16, 2],
            "stages": [
                # [t, c, n, s]
                # stage 0
                [(1, 16, 1, 1)],
                # stage 1
                [(6, 32, 2, 2)],
                # stage 2
                [(6, 48, 3, 2)],
                # stage 3
                [(6, 96, 4, 2), (6, 128, 3, 1)],
                # stage 4, bbox head
                [(4, 128, 1, 2), (6, 128, 2, 1), (6, 160, 1, 1)],
                # stage 5, mask head
                [(4, 128, 1, 2), (6, 128, 2, 1), (6, 128, 1, -2), (3, 64, 1, -2)],
                # stage 6, rpn head
                [(6, 128, 3, 1)],
            ],
            # [c, channel_scale]
            "last": (0, 0.0),
            "backbone": [0, 1, 2, 3],
            "rpn": [6],
            "bbox": [4],
            "mask": [5],
            "kpts": []
        },
    },
    "mobilenet_v2": {
        "stage_op_types": [
            # stage 0
            ["ir_k3"],
            # stage 1
            ["ir_k3"] * 2,
            # stage 2
            ["ir_k3"] * 3,
            # stage 3
            ["ir_k3"] * 7,
            # stage 4
            ["ir_k3"] * 4,
        ],
        "arch_stages": {
            "first": [32, 2],
            "stages": [
                # [t, c, n, s]
                # stage 0
                [(1, 16, 1, 1)],
                # stage 1
                [(6, 24, 2, 2)],
                # stage 2
                [(6, 32, 3, 2)],
                # stage 3
                [(6, 64, 4, 2), (6, 96, 3, 1)],
                # stage 4
                [(6, 160, 3, 1), (6, 320, 1, 1)],
            ],
            # [c, channel_scale]
            "last": (0, 0.0),
            "backbone": [0, 1, 2, 3],
            "bbox": [4],
            "mask": [],
            "kpts": []
        },
    },
}

MODEL_ARCH_CHAM: dict[str, ModelArch] = {
    "cham_v1a": {
        "stage_op_types": [
            # stage 0
            ["ir_k3"],
            # stage 1
            ["ir_k7"] * 2,
            # stage 2
            ["ir_k3"] * 5,
            # stage 3
            ["ir_k5"] * 7 + ["ir_k3"] * 5,
            # stage 4, bbox head
            ["ir_k3"] * 5,
            # stage 5, rpn
            ["ir_k3"] * 3,
        ],
        "arch_stages": {
            "first": [32, 2],
            "stages": [
                # [t, c, n, s]
                # stage 0
                [(1, 24, 1, 1)],
                # stage 1
                [(4, 48, 2, 2)],
                # stage 2
                [(7, 64, 5, 2)],
                # stage 3
                [(12, 56, 7, 2), (8, 88, 5, 1)],
                # stage 4, bbox head
                [(7, 152, 4, 2), (10, 104, 1, 1)],
                # stage 5, rpn head
                [(8, 88, 3, 1)],
            ],
            # [c, channel_scale]
            "last": (0, 0.0),
            "backbone": [0, 1, 2, 3],
            "rpn": [5],
            "bbox": [4],
            "mask": [],
            "kpts": []
        },
    },
    "cham_v2": {
        "stage_op_types": [
            # stage 0
            ["ir_k3"],
            # stage 1
            ["ir_k5"] * 4,
            # stage 2
            ["ir_k7"] * 6,
            # stage 3
            ["ir_k5"] * 3 + ["ir_k3"] * 6,
            # stage 4, bbox head
            ["ir_k3"] * 7,
            # stage 5, rpn
            ["ir_k3"] * 1,
        ],
        "arch_stages": {
            "first": [32, 2],
            "stages": [
                # [t, c, n, s]
                # stage 0
                [(1, 24, 1, 1)],
                # stage 1
                [(8, 32, 4, 2)],
                # stage 2
                [(5, 48, 6, 2)],
                # stage 3
                [(9, 56, 3, 2), (6, 56, 6, 1)],
                # stage 4, bbox head
                [(2, 160, 6, 2), (6, 112, 1, 1)],
                # stage 5, rpn head
                [[6, 56, 1, 1]],
            ],
            # [c, channel_scale]
            "last": (0, 0.0),
            "backbone": [0, 1, 2, 3],
            "rpn": [5],
            "bbox": [4],
            "mask": [],
            "kpts": []
        },
    },
}

for x in MODEL_ARCH_CHAM:
    assert x not in MODEL_ARCH, f"Duplicated model name {x} existed"
    MODEL_ARCH[x] = MODEL_ARCH_CHAM[x]
