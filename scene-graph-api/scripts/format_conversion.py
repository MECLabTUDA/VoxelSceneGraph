import json
import logging
import shutil

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

from logging import getLogger
from pathlib import Path

import click
import click_logging
from tqdm import tqdm

try:
    # If we have scene_graph_annotation installed, find out how the annotation progress file is called to ignore it
    from scene_graph_annotation.utils.progress import CohortAnnotationProgress

    IGNORE_FILE = CohortAnnotationProgress.FILE
except ImportError:
    IGNORE_FILE = ""

from scene_graph_api.knowledge import KnowledgeGraph
from scene_graph_api.scene import SceneGraph
from scene_graph_api.utils.image_utils import supported_extensions_by_type, default_extensions_by_type
from scene_graph_api.utils.nifti_io import NiftiImageWrapper
from scene_graph_api.utils.pathing import remove_suffixes

try:
    from scene_graph_api.tensor_structures import BoxList, BoxListConverter

    boxlist_enabled = True
except ImportError:
    BoxList = BoxListConverter = None
    boxlist_enabled = False

_support_formats = ["labelmap", "segmentation", "SceneGraph"] + (["BoxList"] if boxlist_enabled else [])


@click.command()
@click.option("-k", "--knowledge-path",
              type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path),
              required=True, help="Path to the knowledge graph.")
@click.option("-i", "--input-folder",
              type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path),
              required=True, help="Path to the annotation folder containing the Scene Graphs.")
@click.option("-o", "--output-folder",
              type=click.Path(exists=False, file_okay=False, dir_okay=True, writable=True, path_type=Path),
              required=True, help="Path to the output folder where targets will be saved.")
@click.option("-if", "--input-format",
              type=click.Choice(_support_formats),
              required=True, help="Format of the input data.")
@click.option("-of", "--output-format",
              type=click.Choice(_support_formats),
              required=True, help="Format of the output data.")
def main(knowledge_path: Path, input_folder: Path, output_folder: Path, input_format: str, output_format: str) -> int:
    """
    Convert the annotation for an image from a format to another.
    Supported formats are SceneGraph, labelmap, semantic segmentation, BoxList.
    """
    logger = getLogger(__file__)
    click_logging.basic_config(logger)
    logger.setLevel(logging.INFO)

    # Check that BoxList are enabled if any format is BoxList
    if input_format == "BoxList" or output_format == "BoxList":
        if not boxlist_enabled:
            logger.error("Either the input or the output format is BoxList, "
                         "but the BoxList type could not be imported.")
            return -1

    # Load knowledge
    knowledge_graph = KnowledgeGraph.load(knowledge_path.as_posix(), logger)
    if knowledge_graph is None:
        logger.error("The knowledge could not be read or is not valid.")
        return -1

    # Find all annotation files
    # Figure out which file extensions we can expect
    match input_format:
        case "segmentation" | "labelmap":
            supported_extensions = supported_extensions_by_type[knowledge_graph.get_graph_type()]
        case "SceneGraph":
            supported_extensions = [".json"]
        case "BoxList":
            supported_extensions = [".pth"]
        case _:
            logger.error("Unknown input format. Click should have caught that...")
            return -1

    vol_output_ext = default_extensions_by_type[knowledge_graph.get_graph_type()]
    paths: list[Path] = [
        p
        for p in input_folder.iterdir()
        if "".join(p.suffixes) in supported_extensions and p.name != IGNORE_FILE
    ]

    # Create output folder and save
    output_folder.mkdir(exist_ok=True)

    # Check if this is a no-op
    if input_format == output_format:
        if input_folder.absolute() != output_folder.absolute():
            logger.info(f"Starting the processing:")
            for path in tqdm(paths):
                shutil.copy(path, output_folder / path.name)
        return 0

    # Do the conversion
    logger.info(f"Starting the processing:")

    # TODO ids as contiguous when saving
    if input_format == "BoxList":
        # These conversions are faster
        for path in tqdm(paths):
            boxlist = BoxList.load(path)
            vol_out_path = output_folder / (remove_suffixes(path) + vol_output_ext)
            match output_format:
                case "SceneGraph":
                    BoxListConverter.to_scene_graph(boxlist, knowledge_graph).save(
                        output_folder / (remove_suffixes(path) + ".json")
                    )
                case "labelmap":
                    labelmap, ids_to_class = BoxListConverter.to_labelmap(boxlist)
                    labelmap.save(vol_out_path)
                    with open(output_folder / (remove_suffixes(path) + ".json"), "w") as f:
                        json.dump(ids_to_class, f)
                case "segmentation":
                    BoxListConverter.to_semantic_segmentation(boxlist).save(vol_out_path)
                case _:
                    raise RuntimeError(f"Unsupported output format ({output_format})")
        return 0

    error = False
    for path in tqdm(paths):
        # First load as SceneGraph (or convert from mask) and then convert it to output format
        match input_format:
            case "SceneGraph":
                graph = SceneGraph.load(path.as_posix(), knowledge_graph, logger)
            case "labelmap":
                labelmap = NiftiImageWrapper.load_depth_first(path)
                # Load the class mapping
                mapping_path = path.with_name(remove_suffixes(path) + ".json")
                if not mapping_path.exists():
                    logger.error(f"Could not find the id mapping file {mapping_path.name}")
                    error = True
                    continue
                with mapping_path.open("r") as f:
                    ids_to_class: dict = json.load(f)
                # Assert that the mapping only contains integers
                # noinspection PyTypeChecker
                if not all(map(str.isdigit, key) for key in ids_to_class.keys()):
                    logger.error(f"The mapping {mapping_path.name} contains keys that are not integers.")
                    error = True
                    continue
                if not all(isinstance(val, int) for val in ids_to_class.values()):
                    logger.error(f"The mapping {mapping_path.name} contains values that are not integers.")
                    error = True
                    continue
                # Then cast keys
                ids_to_class = {int(key): val for key, val in ids_to_class.items()}

                try:
                    graph = SceneGraph.create_fom_labelmap(knowledge_graph, labelmap, ids_to_class, logger)
                except Exception as e:
                    logger.error(f"An exception occurred, when converting the labelmap ({mapping_path.name}): {e}")
                    error = True
                    continue
            case "segmentation":
                segmentation = NiftiImageWrapper.load_depth_first(path)
                graph = SceneGraph.create_from_segmentation(knowledge_graph, segmentation, logger)
            case _:
                raise RuntimeError(f"Unsupported input format ({input_format})")

        # Check that the creation/loading worked
        if graph is None:
            logger.error(f"The scene graph ({path.name}) could not be read or is not valid.")
            error = True
            continue

        # Convert
        vol_out_path = output_folder / (remove_suffixes(path) + vol_output_ext)
        match output_format:
            case "SceneGraph":
                graph.save(output_folder / (remove_suffixes(path) + ".json"))
            case "BoxList":
                BoxListConverter(BoxList).from_scene_graph(graph).save(output_folder / (remove_suffixes(path) + ".pth"))
            case "labelmap":
                labelmap, ids_to_class = graph.to_labelmap()
                labelmap.save(vol_out_path)
                with open(output_folder / (remove_suffixes(path) + ".json"), "w") as f:
                    json.dump(ids_to_class, f)
            case "segmentation":
                graph.to_semantic_segmentation().save(vol_out_path)

    return -1 if error else 0


if __name__ == "__main__":
    exit(main())
