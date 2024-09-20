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

import nibabel as nib
import numpy as np
from PIL import Image
from nibabel.filebasedimages import ImageFileError

from scene_graph_api.knowledge.KnowledgeGraph import RadiologyImageKG, NaturalImageKG, KnowledgeGraph
from scene_graph_api.utils.nifti_io import NiftiImageWrapper

# Supported file types for each knowledge graph type
supported_extensions_by_type = {
    RadiologyImageKG.get_graph_type(): [
        file_suffix + compressed_suffix
        for file_suffix in [".nii", ".hdr", ".mnc", ".mgh"]
        for compressed_suffix in ['', '.gz', '.bz2', '.zst']
    ],
    NaturalImageKG.get_graph_type(): [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"]
}

# Sometimes we don't know which type to use since we're doing batch processing and the user has not specified any ext.
default_extensions_by_type = {
    RadiologyImageKG.get_graph_type(): ".nii.gz",
    NaturalImageKG.get_graph_type(): ".png"
}


def get_image_paths(graph: KnowledgeGraph, img_folder_path: Path) -> list[Path]:
    """Given a root path, returns the path to files with a supported format."""
    supported_extensions = supported_extensions_by_type.get(graph.get_graph_type(), [])
    return [p for p in img_folder_path.iterdir() if "".join(p.suffixes) in supported_extensions]


def load_image(
        graph: KnowledgeGraph,
        path: Path,
        is_segmentation: bool = False,
        logger: Logger | None = None
) -> NiftiImageWrapper | None:
    """
    Load an image using the appropriate library (chosen based on the knowledge graph type).
    Can raise exceptions, except when a logger is supplied.
    """
    match graph:
        case RadiologyImageKG():
            try:
                return NiftiImageWrapper.load_depth_first(path)
            except FileNotFoundError as e:
                if logger is not None:
                    logger.error(f"{path.as_posix()}: File not found.")
                    return
                else:
                    raise e
            except ImageFileError as e:
                if logger is not None:
                    logger.error(f"{path.as_posix()}: image could not be read: {e}")
                    return
                else:
                    raise e
        case NaturalImageKG():
            try:
                # RGBA array | None
                array = Image.open(path)
                if not is_segmentation:
                    array = array.convert("RGBA")
                return NiftiImageWrapper(nib.Nifti1Image(np.asarray(array), np.eye(4)), True)
            except Exception as e:
                if logger is not None:
                    logger.error(f"Unexpected exception encountered during loading of {path}: {e}")
                else:
                    raise e
        case _:
            if logger is not None:
                logger.error(f"Unexpected knowledge graph type {type(graph).__name__} during loading of {path}")
