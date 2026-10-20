"""Delete an existing study if found."""
from __future__ import annotations

from logging import getLogger
from pathlib import Path

import click
import click_logging

from scene_graph_database.scripts import study_delete


@click.command()
@click.option("-d", "--db-string",
              type=click.Path(True, True, False, True, True, path_type=Path),
              required=True, help="Path to the database.")
@click.option("-s", "--study-id", type=int, multiple=True, required=True, help="Id of the study.")
def main(db_string: Path, study_id: list[int]) -> int:
    logger = getLogger(__file__)
    click_logging.basic_config(logger)
    logger.setLevel("DEBUG")

    return study_delete(db_string, study_id, logger)


if __name__ == "__main__":
    exit(main())
