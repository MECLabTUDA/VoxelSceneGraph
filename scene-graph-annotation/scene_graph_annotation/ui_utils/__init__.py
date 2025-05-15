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

from PyQt6.QtWidgets import QSizePolicy as _QSizePolicy, QHBoxLayout as _QHBoxLayout, QFrame as _QFrame

from .QAlignRightButton import QAlignRightButton
from .QCollapsibleBox import QCollapsibleBox
from .QGraphView import QGraphView
from .QPixmapLabelWithMouseTracking import QPixmapLabelWithMouseTracking
from .QSelectPathButton import QSelectPathButton
from .QSelectPathWidget import QSelectFolderWidget, QSelectJsonFileWidget


# noinspection SpellCheckingInspection
def hbox_layout_with_vertical_line_left() -> _QHBoxLayout:
    """
    Returns a HBox layout with a vertical line on the left and a small margin, see CollapsibleObjectEditorWidget.
    Utils function defined to avoid duplicate code and have a harmonized look.
    """
    layout = _QHBoxLayout()
    layout.setContentsMargins(18, 0, 0, 0)
    layout.setSpacing(0)

    v_line = _QFrame()
    v_line.setFrameShape(_QFrame.Shape.VLine)
    v_line.setFrameShadow(_QFrame.Shadow.Sunken)
    v_line.setSizePolicy(_QSizePolicy.Policy.Minimum, _QSizePolicy.Policy.MinimumExpanding)
    layout.addWidget(v_line)

    return layout
