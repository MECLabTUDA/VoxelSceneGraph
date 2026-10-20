"""Rebuild the database and delete all existing content."""
from logging import getLogger
from pathlib import Path

import click
import click_logging

from scene_graph_database.scripts import database_rebuild


@click.command()
@click.option("-d", "--db-string",
              type=click.Path(False, True, False, True, True, path_type=Path),
              required=True, help="Path to the database.")
def main(db_string: Path) -> int:
    logger = getLogger(__file__)
    click_logging.basic_config(logger)
    logger.setLevel("DEBUG")

    return database_rebuild(db_string, logger)


if __name__ == "__main__":
    exit(main())
