from functools import wraps
from pathlib import Path
from typing import Callable

import click


def setup_parsing_options(main_func: Callable) -> Callable:
    @click.command()
    @click.option("-k", "--knowledge-path",
                  type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path),
                  required=True, help="Path to the template file.")
    @click.option("-g", "--scene-graph-path",
                  type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path),
                  required=True, help="Path to the scene graph file.")
    @click.option("-h", "--height",
                  type=str, default="800px",
                  required=False, help="Height of the graph canvas. Should have the form \"500px\" or \"50%\".")
    @click.option("-w", "--width",
                  type=str, default="100%",
                  required=False, help="Width of the graph canvas. Should have the form \"500px\" or \"50%\".")
    @click.option("--show-buttons",
                  type=click.Choice(["nodes", "edges", "physics"], case_sensitive=True),
                  default=None, multiple=True,
                  required=False, help='None, one or multiple of ["nodes", "edges", "physics"].')
    @click.option("--node-font-size",
                  type=int, default=None,
                  required=False, help="Node font size in px.")
    @click.option("--edge-font-size",
                  type=int, default=None,
                  required=False, help="Node font size in px.")
    @click.option("--graph-options",
                  type=str, default=None,
                  required=False, help="CSS options as a string representing the JS object with options "
                                       "(see the doc in the VisJS framework).")
    @click.option("-o", "--output",
                  type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True, path_type=Path),
                  required=True, help="Path to the HTML output file.")
    @wraps(main_func)
    def wrapper_common_options(*args, **kwargs):
        return main_func(*args, **kwargs)

    return wrapper_common_options
