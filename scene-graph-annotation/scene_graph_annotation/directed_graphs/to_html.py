"""
pyvis is used to convert networkx graphs to html. vis.js handles the rest in javascript.

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
from typing import Literal

try:
    from networkx import MultiDiGraph
    from pyvis.network import Network
except ImportError:
    MultiDiGraph = None
    Network = None


def networkx_to_html(
        graph: MultiDiGraph,
        height: str = "100%",
        width: str = "100%",
        show_buttons: list[Literal["nodes", "edges", "physics"]] | None = None,
        node_font_size: int | None = None,
        edge_font_size: int | None = None,
        graph_options: str | None = None
) -> str:
    """
    Converts a networkx directed multi-graph to html using pyvis.
    Note: Only use height and width in pixels in you wish to render the graph in a widget,
          as it allows for easy <div> resize when the network size is changed in javascript.
          For opening the graph in a browser, I recommend leaving the width to 100% and to set the height in pixels.
    Note: vis.js filter_menu is always on and js lib resources are inlined.
    :param graph: the graph to display.
    :param show_buttons: optional list of buttons to display. Only works in a browser.
                         the layout in the annotator does not allow for this kind of customization.
    :param height: height as string in px or % e.g. "1000px" pr "100%".
    :param width: same as for the height.
    :param node_font_size: optional node font size. Set to 20 or more for a bigger text.
    :param edge_font_size: optional edge font size. Set to 20 or more for a bigger text.
    :param graph_options: optional CSS options as a str representation of the JS object.
                          it will override any other parameter.
                          see doc in the VisJS framework.
    """

    nt = Network(height, width, filter_menu=True, directed=True, cdn_resources="in_line")
    nt.from_nx(graph)
    nt.inherit_edge_colors(False)

    if show_buttons is not None:
        nt.show_buttons(filter_=show_buttons)

    if node_font_size is not None:
        for n in nt.nodes:
            n["font"] = {"size": node_font_size}
    if edge_font_size is not None:
        for e in nt.edges:
            e["font"] = {"size": edge_font_size}

    if graph_options is not None:
        nt.set_options(graph_options)

    # Replace <div> class "card-body" to remove some annoying padding
    return nt.generate_html().replace('class="card-body"', 'class="other"')
