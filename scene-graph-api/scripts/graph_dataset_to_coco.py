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

import json
from logging import getLogger
from pathlib import Path

import click
import click_logging
from pycocotools3d.coco.abstractions.common import Image

from scene_graph_api.knowledge import KnowledgeGraph
from scene_graph_api.scene import SceneGraph

_supported_extensions = [
    file_suffix + compressed_suffix
    for file_suffix in [".nii", ".hdr", ".mnc", ".mgh"]
    for compressed_suffix in ['', '.gz', '.bz2', '.zst']
]


@click.command()
@click.option("-k", "--knowledge-path",
              type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path),
              required=True, help="Path to the knowledge file.")
@click.option("-i", "--image-folder",
              type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path),
              required=True, help="Path to the image folder.")
@click.option("-a", "--annotation-folder",
              type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path),
              required=True, help="Path to the annotation folder containing the Scene Graphs.")
@click.option("-o", "--output",
              type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True, path_type=Path),
              required=True, help="Path to the JSON output file where the COCO-style annotation will be saved.")
def main(template_path: Path, image_folder: Path, annotation_folder: Path, output: Path) -> int:
    logger = getLogger(__file__)
    click_logging.basic_config(logger)

    # Load knowledge
    template = KnowledgeGraph.load(template_path, logger)
    if template is None:
        logger.error("The knowledge could not be read or is not valid.")
        return -1

    # Find all images
    image_paths: list[Path] = [p for p in image_folder.iterdir() if "".join(p.suffixes) in _supported_extensions]
    logger.info(f"Found {len(image_paths)} images.")

    # Find the annotation file corresponding to each image
    missing = False
    graph_paths: list[Path] = []
    for image_path in image_paths:
        graph_path = annotation_folder / (image_path.name.replace("".join(image_path.suffixes), ".json"))
        if not graph_path.exists():
            logger.error(f"Image path ({image_path.as_posix()}) does not have an annotation file "
                         f"({graph_path.as_posix()}).")
            missing = True
        graph_paths.append(graph_path)
    if missing:
        return -1

    # Check that each Scene Graph can be read and is valid
    error = False
    graphs: list[SceneGraph] = []
    for graph_path in graph_paths:
        graph = SceneGraph.load(graph_path, template, logger)
        if graph is None:
            logger.error(f"The scene graph ({graph_path.as_posix()}) could not be read or is not valid.")
            error = True
        graphs.append(graph)
    if error:
        return -1

    # Build COCO annotation
    coco = template.to_coco()
    for idx, (image_path, graph) in enumerate(zip(image_paths, graphs)):
        # Add image
        n_dim = len(graph.object_labelmap.shape)
        image_dict: Image = {
            "id": idx,
            "file_name": image_path.as_posix(),
            "height": graph.object_labelmap.shape[-2],
            "width": graph.object_labelmap.shape[-1],
        }
        if n_dim == 3:
            image_dict["depth"] = graph.object_labelmap.shape[-3]

        # Add annotations
        graph.add_coco_annotations(coco, idx)

    # Create output folder and save
    output.mkdir(exist_ok=True)
    with open(output, "w") as f:
        json.dump(coco, f)

    logger.info("All done!")
    return 0


if __name__ == "__main__":
    exit(main())
