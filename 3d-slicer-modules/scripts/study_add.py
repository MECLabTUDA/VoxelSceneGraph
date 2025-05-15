"""Add a new study to the database."""
from __future__ import annotations

import sqlite3
from logging import getLogger
from pathlib import Path

import click
import click_logging

from annotation_database.functions import study_add
from annotation_database.options import Options
from annotation_database.structures import Study


@click.command()
@click.option("-d", "--db-string",
              type=click.Path(True, True, False, True, True, path_type=Path),
              required=True, help="Path to the database.")
@click.option("-n", "--name", type=str, required=True, help="Name of the study.")
@click.option("-i", "--image-folder", type=click.Path(path_type=Path),
              required=True, help="Relative path to the image folder from the database path.")
@click.option("-l", "--label-folder", type=click.Path(path_type=Path),
              required=True, help="Relative path to the label folder from the database path.")
@click.option("-s", "--segments", type=str, multiple=True, required=True, help="List of segment names.")
@click.option("-r", "--last-segment-can-repeat", type=bool,
              is_flag=True, default=False, help="Whether the last segment can repeat.")
@click.option("--window-width", type=int, required=False, default=None, help="Window width for display.")
@click.option("--window-center", type=int, required=False, default=None, help="Window center for display.")
def main(
        db_string: Path,
        name: str,
        image_folder: Path,
        label_folder: Path,
        segments: list[str],
        last_segment_can_repeat: bool,
        window_width: int | None,
        window_center: int | None
) -> int:
    logger = getLogger(__file__)
    click_logging.basic_config(logger)

    # First try to get access to the db
    Options.set_db_string(db_string.as_posix())
    try:
        Options.get_db()
    except sqlite3.OperationalError as e:
        logger.error(f"Could not access database: {e}")
        return 1

    # Build the study object
    study = Study(
        id=None,  # Auto filled
        name=name,
        img_folder=image_folder,
        label_folder=label_folder,
        segments=segments,
        last_segment_can_repeat=last_segment_can_repeat,
        progress={},
        window_width=window_width,
        window_center=window_center
    )
    study_add(study)

    print(f"Added the study (id={study.id}).")
    return 0


if __name__ == "__main__":
    exit(main())
