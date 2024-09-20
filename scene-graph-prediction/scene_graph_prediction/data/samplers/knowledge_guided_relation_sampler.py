import torch
from yacs.config import CfgNode

from scene_graph_api.tensor_structures import BoxList
from scene_graph_prediction.data.datasets import COCOEvaluableDataset
from scene_graph_prediction.structures import ImageList
from .knowledge_guided_abstract_sampler import KnowledgeGuidedAbstractSampler


class KnowledgeGuidedRelationSampler(KnowledgeGuidedAbstractSampler):
    """
    Similar to a GroupedBatchSampler fused with a RandomSampler, but we sequentially select a group for a fixed number
    of iterations and directly sample images with objects containing objects in this group.
    Groups are the foreground relation classes.
    If strict=True, we only want to train to detect relations within the currently selected group.

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
        assert cfg.MODEL.RELATION_ON, "cfg.MODEL.RELATION_ON is required for this sampler to work."
        assert cfg.INPUT.N_REL_CLASSES > 2, "This sampler requires at least two foreground relation classes."

        super().__init__(
            cfg=cfg,
            batch_size=batch_size,
            n_groups=cfg.INPUT.N_REL_CLASSES - 1,
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
            relations = target.RELATIONS

            for rel_idx in torch.unique(relations):
                if rel_idx.item() == 0:
                    continue
                group_to_ids[rel_idx.item()].append(img_idx)

        return group_to_ids

    def batch_collation(self, batch: list) -> tuple[ImageList, tuple[BoxList, ...], tuple[int, ...]]:
        images, targets, img_ids = super().batch_collation(batch)

        if self.strict_sampling:
            # Mark relevant foreground relations as ignored
            for target in targets:
                # Note: we don't have to make a copy
                target.RELATIONS[(target.RELATIONS > 0) & (target.RELATIONS != self.current_group + 1)] = -1

        return images, targets, img_ids
