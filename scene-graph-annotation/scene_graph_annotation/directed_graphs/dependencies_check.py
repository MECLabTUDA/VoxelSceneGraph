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


def check_graph_plot_dependencies() -> tuple[bool, list[Exception]]:
    """
    Check that the libraries required for plotting directed graphs are installed.
    :returns: success, list of errors of any sort (import or otherwise)
    """
    success = True
    errors = []

    try:
        import networkx
    except Exception as e:
        success = False
        errors.append(e)

    try:
        import pyvis
    except Exception as e:
        success = False
        errors.append(e)

    return success, errors


def check_graph_render_dependencies() -> tuple[bool, list[Exception]]:
    """
    Check that the libraries required for rendering directed graphs are installed.
    :returns: success, list of errors of any sort (import or otherwise)
    """
    success = True
    errors = []

    try:
        from PyQt6.QtWebEngineWidgets import QWebEngineView
    except Exception as e:
        success = False
        errors.append(e)

    try:
        from PyQt6.QtWebEngineCore import QWebEngineSettings
    except Exception as e:
        success = False
        errors.append(e)

    return success, errors
