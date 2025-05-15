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

import re
from logging import getLogger
from pathlib import Path
from typing import Literal

import click_logging

from scene_graph_annotation.directed_graphs import dependencies_check, to_networkx, to_html
from scene_graph_annotation.directed_graphs.script_utils import setup_parsing_options
from scene_graph_annotation.knowledge import KnowledgeGraph
from scene_graph_annotation.scene import SceneGraph


# noinspection DuplicatedCode
@setup_parsing_options
def main(
        knowledge_path: Path,
        scene_graph_path: Path,
        height: str,
        width: str,
        show_buttons: list[Literal["nodes", "edges", "physics"]] | None,
        node_font_size: int | None,
        edge_font_size: int | None,
        graph_options: str | None,
        output: Path
) -> int:
    logger = getLogger(__file__)
    click_logging.basic_config(logger)

    # Dependencies check
    success, errors = dependencies_check.check_graph_plot_dependencies()
    if not success:
        logger.error("Following errors occurred when loading required libraries")
        for error in errors:
            logger.error(str(error))
        return -1

    # Load template
    knowledge_graph = KnowledgeGraph.load(knowledge_path, logger)
    if knowledge_graph is None:
        logger.error("The knowledge graph could not be read or is not valid.")
        return -1

    # Load scene graph
    scene_graph = SceneGraph.load(scene_graph_path, knowledge_graph, logger)
    if scene_graph is None:
        logger.error("The scene graph could not be read or is not valid.")
        return -1

    # Try to parse height and width
    length_regex = r"^\d+(?:(?:px)|%)$"
    if not re.match(length_regex, height):
        logger.error(f"Invalid height {height}. Please check the format.")
        return -1
    if not re.match(length_regex, width):
        logger.error(f"Invalid width {width}. Please check the format.")
        return -1

    # Find the annotation file corresponding to each image
    knowledge_graph = to_networkx.scene_graph_to_networkx(scene_graph)
    html = to_html.networkx_to_html(
        knowledge_graph,
        height=height,
        width=width,
        show_buttons=show_buttons,
        node_font_size=node_font_size,
        edge_font_size=edge_font_size,
        graph_options=graph_options
    )

    with output.open("w", encoding="utf-8") as f:
        f.write(html)

    logger.info("All done!")
    return 0


if __name__ == "__main__":
    exit(main())
