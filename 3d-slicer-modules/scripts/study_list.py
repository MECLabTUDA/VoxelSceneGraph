"""List all studies."""
from __future__ import annotations

import sqlite3
from logging import getLogger
from pathlib import Path

import click
import click_logging

from annotation_database.options import Options
from annotation_database.functions import study_get_all


@click.command()
@click.option("-d", "--db-string",
              type=click.Path(True, True, False, True, True, path_type=Path),
              required=True, help="Path to the database.")
def main(db_string: Path) -> int:
    logger = getLogger(__file__)
    click_logging.basic_config(logger)

    # First try to get access to the db
    Options.set_db_string(db_string.as_posix())
    try:
        Options.get_db()
    except sqlite3.OperationalError as e:
        logger.error(f"Could not access database: {e}")
        return 1

    studies = study_get_all(False)
    print(f"{len(studies)} stud{'y' if len(studies) < 2 else 'ies'} found:")
    for study in studies.values():
        print(f"\t{study}")

    return 0


if __name__ == "__main__":
    exit(main())
