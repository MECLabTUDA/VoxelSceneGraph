import torch
from yacs.config import CfgNode

from scene_graph_api.tensor_structures import BoxList
from scene_graph_prediction.data.datasets import COCOEvaluableDataset
from scene_graph_prediction.structures import ImageList
from .knowledge_guided_abstract_sampler import KnowledgeGuidedAbstractSampler


class KnowledgeGuidedObjectSampler(KnowledgeGuidedAbstractSampler):
    """
    Similar to a GroupedBatchSampler fused with a RandomSampler, but we sequentially select a group for a fixed number
    of iterations and directly sample images with objects containing objects in this group.
    Groups are expected to be annotated as an N-sized binary ATTRIBUTES tensor.
    The presence of relevant object within an image should be annotated as an N-sized binary IMAGE_ATTRIBUTES tensor.
    Objects may belong to multiple groups.
    If strict=True, we only want to train to detect objects within the currently selected group.
    Since the RPN, (one- / two-stage) detector look at different fields,
    the easiest solution is to set the weight of other boxes to 0. This requires weighted training.

    WARNING: this Sampler loops forever, so it has to be used with an IterationBasedBatchSampler.
    """

    def __init__(
            self,
            cfg: CfgNode,
            batch_size: int,
            iterations_per_group: int,
            num_batches: int,
            start_batch: int,
            strict_sampling: bool = False,
            group_to_ids: dict[int, list[int]] | None = None  # Mostly for testing purposes
    ):
        assert not cfg.MODEL.RELATION_ON, "cfg.MODEL.RELATION_ON cannot be True for this sampler to work."
        assert cfg.MODEL.WEIGHTED_BOX_TRAINING, "cfg.MODEL.WEIGHTED_BOX_TRAINING is required for this sampler to work."
        assert cfg.INPUT.N_IMG_ATT_CLASSES > 1, "This sampler requires at least two object attribute classes"
        assert not strict_sampling or cfg.INPUT.N_ATT_CLASSES == cfg.INPUT.N_IMG_ATT_CLASSES, \
            "This sampler requires as many object attributes as image attributes for strict sampling."

        super().__init__(
            cfg=cfg,
            batch_size=batch_size,
            n_groups=cfg.INPUT.N_IMG_ATT_CLASSES,
            iterations_per_group=iterations_per_group,
            num_batches=num_batches,
            start_batch=start_batch,
            strict_sampling=strict_sampling,
            group_to_ids=group_to_ids
        )

    def _compute_group_ids_mapping(self, dataset: COCOEvaluableDataset) -> dict[int, list[int]]:
        """Use the image attributes to determine groups. An image can be in multiple groups."""
        group_to_ids = {group_id: [] for group_id in range(self.n_groups)}

        for img_idx in range(len(dataset)):
            target = dataset.get_groundtruth(img_idx)
            img_attr = target.IMAGE_ATTRIBUTES

            for attr_idx in torch.nonzero(img_attr):
                group_to_ids[attr_idx.item()].append(img_idx)

        return group_to_ids

    def batch_collation(self, batch: list) -> tuple[ImageList, tuple[BoxList, ...], tuple[int, ...]]:
        images, targets, img_ids = super().batch_collation(batch)

        if self.strict_sampling:
            # Set the weights of objects not belonging to the group to 0
            for target in targets:
                # Note: we don't have to make a copy
                target.IMPORTANCE[target.ATTRIBUTES[:, self.current_group] == 0] = 0.

        return images, targets, img_ids
