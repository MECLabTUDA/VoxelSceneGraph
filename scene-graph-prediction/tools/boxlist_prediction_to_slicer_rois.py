"""
Given the path to an input image and a BoxList (predictions or ground truth),
prints code to copy in 3D Slicer to display the bounding boxes.

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

import logging
from pathlib import Path

import click
import click_logging
import numpy as np
import torch
from scene_graph_api.knowledge import KnowledgeGraph

from scene_graph_prediction.structures import BoxList


@click.command()
@click.option("-b", "--boxlist-path",
              type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path),
              required=True, help="Path to the bounding boxes (as BoxList) that will be displayed.")
@click.option("-k", "--knowledge-graph-path",
              type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path),
              required=False, help="Optional: Path to the knowledge graph (used to use the same color palette).")
@click.option("-g", "--ground-truth", is_flag=True,
              help="Whether the bounding boxes are ground truth or predictions. "
                   "When unspecified, the latter is assumed.")
@click.option("-n", "--top-n",
              type=int, default=0,
              required=False, help="Maximum number of bounding boxes to display.")
@click.option("-c", "--copy", is_flag=True,
              help="Whether the produced Python code should be copied to the clipboard rather than being printed. "
                   "This may not work on headless systems.")
def main(
        boxlist_path: Path,
        knowledge_graph_path: Path | None = None,
        ground_truth=False,
        top_n: int = 0,
        copy: bool = False
) -> int:
    # Load data
    logger = logging.getLogger(__name__)
    click_logging.basic_config(logger)
    pred = BoxList.load(boxlist_path)
    boxes = pred.convert(BoxList.Mode.zyxdhw).boxes

    # Get the correct fields
    if ground_truth:
        labels = pred.get_field(BoxList.AnnotationField.LABELS)
        if labels is None:
            print("The BoxList is missing the field \"LABELS\". Are you sure that these are groundtruth annotations?")
            return -1
        else:
            labels = labels.int()
        scores = torch.ones(labels.numel())
    else:
        labels = pred.get_field(BoxList.PredictionField.PRED_LABELS)
        scores = pred.get_field(BoxList.PredictionField.PRED_SCORES)
        if labels is None:
            print("The BoxList is missing the field \"PRED_LABELS\". "
                  "Are you sure that these are predictions?")
            return -1
        else:
            labels = labels.int()
        if scores is None:
            print("The BoxList is missing the field \"PRED_SCORES\". "
                  "Are you sure that these are predictions?")
            return -1

    # Load color palette
    if knowledge_graph_path is not None:
        logger = logging.getLogger(__file__)
        click_logging.basic_config(logger)
        knowledge = KnowledgeGraph.load(knowledge_graph_path.as_posix(), logger)
        if knowledge is None:
            print("Error: the knowledge graph could not be read. Try again without specifying it in the command.")
            return 1
        # Hex-color string to float
        hex_colors = {c.id: c.color for c in knowledge.classes}
        int_colors = {i: (int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)) for i, c in hex_colors.items()}
        float_colors = {i: (c[0] / 255, c[1] / 255, c[2] / 255) for i, c in int_colors.items()}
    else:
        float_colors = {}

    # Load affine matrix, mp.eye if missing
    if pred.has_field(BoxList.AnnotationField.AFFINE_MATRIX):
        affine_matrix = pred.AFFINE_MATRIX.numpy()
    else:
        affine_matrix = np.eye(4)

    print_str = "import numpy as np\n"
    print_str += "from numpy import float32\n"

    # Compute image origin in real-world coordinates
    # Converting the bbox coordinates to real-world coordinates does not work because the ROIs do not get rotated
    # Instead, we create ROIs with the matrix coordinates and apply a linear transformation
    # (containing the image's affine matrix) in 3D Slicer

    # Create the transform and set the affine matrix

    # Important Note:
    # Applying an affine transformation to the ROI has the benefit of also setting the orientation of the ROI
    # Otherwise, the ROIs will not be orientated properly...

    print_str += 'transform_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")\n'
    print_str += f'transform_node.SetName("transform_{boxlist_path.name}")\n'
    print_str += f'np_affine = np.{repr(affine_matrix)}\n'
    print_str += f'slicer.util.updateTransformMatrixFromArray(transform_node, np_affine)\n'

    for idx, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        # Prepare bbox shape
        size: torch.Tensor = box[pred.n_dim: 2 * pred.n_dim]
        center: torch.Tensor = box[:pred.n_dim] + .5 * (size - 1)

        # Prepare strings
        roi_name = f"BBox {idx:2d} Label {label:2d} Score: {score:.3f}"
        # Coordinates need to be depth first
        size_str = ", ".join(f"{s:.3f}" for s in size.tolist())
        center_str = ", ".join(f"{c:.3f}" for c in center.tolist())

        print_str += 'roi = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")\n'
        # Set name
        print_str += f'roi.SetName("{roi_name}")\n'
        # Set as not resizable, with no filling and not visible (to not crowd the image too much)
        print_str += 'roi.GetDisplayNode().SetHandlesInteractive(False)\n'
        print_str += 'roi.GetDisplayNode().FillOpacityOff()\n'
        # Set real-world size for the ROI
        print_str += f'roi.SetSize({size_str})\n'
        print_str += f'roi.SetCenter({center_str})\n'
        print_str += 'roi.SetAndObserveTransformNodeID(transform_node.GetID())\n'
        # Set ROI color to match the knowledge graph (if specified)
        if (color := float_colors.get(label.item(), None)) is not None:
            # Note: Parentheses are added by  the tuple to str conv
            print_str += f'roi.GetDisplayNode().SetSelectedColor{color}\n'

        if 0 < top_n <= idx + 1:
            # Only display top-n bounding boxes
            break

    if copy:
        import pyperclip
        pyperclip.copy(print_str)
        print("Code copied to clipboard!")
    else:
        print(print_str)

    return 0


if __name__ == "__main__":
    exit(main())
