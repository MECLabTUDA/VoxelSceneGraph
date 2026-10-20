"""
Update the progress state for a given study.
For all patients found in the database or the drive:
- Check that the image is present for the patient name, otherwise delete the entry
- If the image is present and no Scene Graph but an AI prediction is present:
create a Scene Graph from the prediction (seg or labelmap)
- If the image is present and no Scene Graph: create an empty Scene Graph
- If there is a Scene Graph but no image: move it to a lost and found folder as provided
Note: we ignore preview images that do not have counterpart images
Note: for matching reasons, we only support "*.nii.gz" file extensions.
"""
from __future__ import annotations

from logging import getLogger
from pathlib import Path

import click
import click_logging

from scene_graph_database.scripts import study_progress_update
from scene_graph_database.structures import AnnotationState


@click.command()
@click.option("-d", "--db-string",
              type=click.Path(True, True, False, True, True, path_type=Path),
              required=True, help="Path to the database.")
@click.option("-s", "--study-id", type=int, required=True, help="Id of the study.")
@click.option("-p", "--progress-new",
              type=click.IntRange(min(AnnotationState).value, max(AnnotationState).value),
              required=False, default=3, help="Progress state for new patient entries with an existing label. " +
                                              ", ".join(f"{state.value}: {state.name}" for state in AnnotationState))
@click.option("-l", "--lost-folder",
              type=click.Path(False, False, True, readable=True, path_type=Path),
              required=True, help="Path to a lost-and-found folder where to move labels wit hno matching image.")
def main(db_string: Path, study_id: int, progress_new: int, lost_folder: Path) -> int:
    logger = getLogger(__file__)
    click_logging.basic_config(logger)
    logger.setLevel("DEBUG")

    return study_progress_update(db_string, study_id, progress_new, lost_folder, logger)


if __name__ == "__main__":
    exit(main())
