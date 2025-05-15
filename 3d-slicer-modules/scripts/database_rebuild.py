"""Rebuild the database and delete all existing content."""
import sqlite3
from logging import getLogger
from pathlib import Path

import click
import click_logging

from annotation_database.functions import delete_tables, init_tables
from annotation_database.options import Options


@click.command()
@click.option("-d", "--db-string",
              type=click.Path(False, True, False, True, True, path_type=Path),
              required=True, help="Path to the database.")
def main(db_string: Path, ) -> int:
    logger = getLogger(__file__)
    click_logging.basic_config(logger)

    # First try to get access to the db
    Options.set_db_string(db_string.as_posix())
    try:
        Options.get_db()
    except sqlite3.OperationalError as e:
        logger.error(f"Could not access database: {e}")
        return 1

    delete_tables()
    init_tables()

    print("Database rebuilt.")
    return 0


if __name__ == "__main__":
    exit(main())
