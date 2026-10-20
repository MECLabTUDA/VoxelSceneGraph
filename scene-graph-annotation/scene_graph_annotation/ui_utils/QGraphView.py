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

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QResizeEvent
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel

from scene_graph_annotation.directed_graphs import dependencies_check, to_networkx, to_html
from scene_graph_annotation.knowledge import KnowledgeGraph
from scene_graph_annotation.scene import SceneGraph


class QGraphView(QWidget):
    """
    Widget holding a QWebEngineView used to display a KnowledgeGraph or Scenegraph in its HTML-graph-form.
    Note: the QWebEngineView is only created on the first view.
    """

    PLOT_DEPENDENCIES_OK, _ = dependencies_check.check_graph_plot_dependencies()
    RENDER_DEPENDENCIES_OK, _ = dependencies_check.check_graph_render_dependencies()

    # noinspection PyMethodParameters
    def __init__(self, object_to_render: KnowledgeGraph | SceneGraph):
        super().__init__()

        self._object_to_render = object_to_render

        self._layout = QHBoxLayout()

        self._engine_view = None
        self._html = None

        self.init_ui()

    def init_ui(self):
        self.setLayout(self._layout)

        # Missing dependencies: just put a placeholder
        if not (self.PLOT_DEPENDENCIES_OK and self.RENDER_DEPENDENCIES_OK):
            placeholder = QLabel("Could not load dependencies to plot and render the graph.")
            self._layout.addWidget(placeholder)
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def on_view(self):
        """Update the graph's HTML representation if necessary."""
        if not (self.PLOT_DEPENDENCIES_OK and self.RENDER_DEPENDENCIES_OK):
            return

        # Set up the engine view if that has not been done before
        if self._engine_view is None:
            from PyQt6.QtWebEngineWidgets import QWebEngineView
            from PyQt6.QtWebEngineCore import QWebEngineSettings

            self._engine_view = QWebEngineView()
            self._layout.addWidget(self._engine_view)
            # Disable scroll bars as although the canvas gets adjusted on resize, bars can still appear for a moment
            self._engine_view.page().settings().setAttribute(QWebEngineSettings.WebAttribute.ShowScrollBars, False)
            # Prepare the callback for resize and fit on load finish
            self._engine_view.page().loadFinished.connect(lambda: self._js_resize(fit=True))

        if self._html is None:
            # Compute the networkx object and the html content
            if isinstance(self._object_to_render, KnowledgeGraph):
                graph = to_networkx.knowledge_graph_to_networkx(self._object_to_render)
            else:
                graph = to_networkx.scene_graph_to_networkx(self._object_to_render)
            self._html = to_html.networkx_to_html(graph, height="100%", width="100%")
            self._engine_view.page().setHtml(self._html)
            self._js_resize(fit=True)

    def clear_graph(self):
        """Method used to clear the graph's HTML representation. It will be computed again on the next view."""
        self._html = None

    def resizeEvent(self, a0: QResizeEvent) -> None:
        # Note: to need for a single shot timer as the QWebEngineView is smart enough
        super().resizeEvent(a0)

        # If we actually have an engine view and the HTML content does not need to be updated
        if self._engine_view is not None and self._html is not None:
            self._js_resize()

    def _js_resize(self, fit: bool = False):
        """
        :param fit: whether we should fit the graph to the current canvas size.
        """
        # Inner div has 1 REM padding along each side
        # 1 REM is equal to the font size of the root element
        # Note: to get the height, we need to remove the height of the filter menu and the fixed borders (2 px)
        # Note: if this is the first call, we need to fit the network to the actual size of the widget
        size = self._engine_view.size()
        self._engine_view.page().runJavaScript(
            f"padding = parseInt(getComputedStyle(document.documentElement).fontSize);"
            f"network.setSize("
            f"  {size.width()} - 4,"  # 2 borders horizontally, 6 borders vertically
            f"  {size.height()} - document.getElementById(\"filter-menu\").offsetHeight - 12"
            f"); "
            f"network.redraw();" +
            ("network.fit();" if fit else "")
        )
