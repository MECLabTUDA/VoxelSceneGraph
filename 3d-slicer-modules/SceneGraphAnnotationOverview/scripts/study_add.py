"""Add a new study to the database."""
from __future__ import annotations

from logging import getLogger
from pathlib import Path

import click
import click_logging

from scene_graph_database.scripts import study_add


@click.command()
@click.option("-d", "--db-string",
              type=click.Path(True, True, False, True, True, path_type=Path),
              required=True, help="Path to the database.")
@click.option("-n", "--name", type=str, required=True, help="Name of the study.")
@click.option("-k", "--knowledge-graph", "knowledge_graph_path",
              type=click.Path(path_type=Path, file_okay=True, dir_okay=False),
              required=True, help="Path to the knowledge graph absolute or relative to the database path.")
def main(
        db_string: Path,
        name: str,
        knowledge_graph_path: Path,
) -> int:
    logger = getLogger(__file__)
    click_logging.basic_config(logger)
    logger.setLevel("DEBUG")

    return study_add(db_string, name, knowledge_graph_path, logger)[1]


if __name__ == "__main__":
    exit(main())
