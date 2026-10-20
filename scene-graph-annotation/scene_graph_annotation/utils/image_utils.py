"""
Copyright 2023 Antoine Sanner, Technical University of Darmstadt, Darmstadt, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from logging import Logger
from pathlib import Path
from typing import Type

import numpy as np
from PIL import Image
from PyQt6.QtGui import QPixmap, QImage

from scene_graph_annotation.knowledge import KnowledgeGraph, RadiologyImageKG, NaturalImageKG
from scene_graph_api.utils.image_utils import get_image_paths, load_image
from .ArrayView import ArrayView, RadiologyImageArrayView, NaturalImageArrayView
from .pathing import remove_suffixes


def rgba_array_to_qpixmap(arr: np.ndarray) -> QPixmap:
    """Converts an RGBA array to a QPixmap."""
    q_image = QImage(arr, arr.shape[1], arr.shape[0], QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(q_image)


def combine_rgba_images(rgba_im1: np.ndarray, rgba_im2: np.ndarray):
    """Combines two RGBA arrays into one, while taking transparency into account."""
    rgba_im = Image.fromarray(rgba_im1)
    rgba_im2 = Image.fromarray(rgba_im2)
    rgba_im.paste(rgba_im2, mask=rgba_im2)
    # noinspection PyTypeChecker
    return np.asarray(rgba_im)


def find_patients_to_annotate(
        knowledge_graph: KnowledgeGraph | Type[KnowledgeGraph],
        img_folder: Path,
        ann_folder: Path,
) -> dict[str, tuple[Path, Path]]:
    """
    Find patients (image found) that eiter has an annotation file.
    Note: the segmentation path is None if the file does not exist.
          the annotation path is not None because we always at least need it for saving the graph later.
    Note: we  do this because we might have deleted raw segmentations,
          but still have the annotations that we want to edit.
    :returns: patient name: (img path, ann path)
    """

    def pat_to_ann_file(patient: str) -> Path:
        return ann_folder / f"{patient}.json"

    if not img_folder.is_dir() or not ann_folder.is_dir():
        return {}

    # Find files and patient name
    pat_name_to_img_path = {remove_suffixes(p): p for p in get_image_paths(knowledge_graph, img_folder)}
    pat_name_to_ann_path = {pat: pat_to_ann_file(pat) for pat in pat_name_to_img_path if pat_to_ann_file(pat).exists()}
    # Find matches (i.e. segmentation or annotation file exists)
    matches = set(pat_name_to_img_path.keys()).intersection(pat_name_to_ann_path.keys())

    # Note: we need to build the path again because the annotation file likely does not exist yet
    return {pat: (pat_name_to_img_path[pat], pat_to_ann_file(pat)) for pat in matches}


def load_array_view(
        knowledge_graph: KnowledgeGraph,
        path: Path,
        logger: Logger
) -> ArrayView | None:
    """Load an image using the right library and return the appropriate ArrayView subclass."""
    image = load_image(knowledge_graph, path, False, logger)
    if image is None:
        return

    match knowledge_graph:
        case RadiologyImageKG():
            # noinspection PyTypeChecker
            return RadiologyImageArrayView(knowledge_graph, image)
        case NaturalImageKG():
            # noinspection PyTypeChecker
            return NaturalImageArrayView(image)
        case _:
            # Same error already logged in scene_graph_api function
            # logger.error(f"Unexpected knowledge graph type {knowledge_graph.__name__} during loading of {path}")
            ...
