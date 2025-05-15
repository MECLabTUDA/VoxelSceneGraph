"""List all files and their progress for a given study."""
from __future__ import annotations

from sqlite_utils.db import NotFoundError


import sqlite3
from logging import getLogger
from pathlib import Path

import click
import click_logging

from annotation_database.options import Options
from annotation_database.functions import study_get


@click.command()
@click.option("-d", "--db-string",
              type=click.Path(True, True, False, True, True, path_type=Path),
              required=True, help="Path to the database.")
@click.option("-s", "--study-id", type=int, required=True, help="Id of the study.")
def main(db_string: Path, study_id: int) -> int:
    logger = getLogger(__file__)
    click_logging.basic_config(logger)

    # First try to get access to the db
    Options.set_db_string(db_string.as_posix())
    try:
        Options.get_db()
    except sqlite3.OperationalError as e:
        logger.error(f"Could not access database: {e}")
        return 1

    try:
        study = study_get(study_id, True)
        print(f"Found {len(study.progress)} files for study (id={study_id}, name={study.name}):")
        for patient_name, progress in study.progress.items():
            progress.comment = progress.comment.replace("\n", " ")  # To avoid formatting issues
            print(f"\t{patient_name}: {progress}")

    except NotFoundError:
        logger.error(f"Study {study_id} does not exist.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
