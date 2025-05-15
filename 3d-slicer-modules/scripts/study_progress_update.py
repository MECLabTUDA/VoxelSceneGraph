"""
Update the progress state for a given study.
For patients found in the database:
- Check that the image is present for the patient name, otherwise delete the entry
- If the image is present but no label: create an empty label
- If there is a label but no image: move it to a lost and found folder as provided
For files found on the drive:
- Check that there is both an image and a label, then create a new entry with the provided progress state
- If the image is present but there is no label: create an empty label and create a new entry
- If there is a label but no image: move it to a lost and found folder as provided
"""
from __future__ import annotations

import sqlite3
from logging import getLogger
from pathlib import Path
from datetime import datetime

import click
import click_logging
import nibabel as nib
import numpy as np
from sqlite_utils.db import NotFoundError

from annotation_database.functions import study_get, progress_add_to_study, INIT_READER, progress_delete_from_study, \
    progress_update_to_study
from annotation_database.options import Options
from annotation_database.structures import AnnotationState, AnnotationProgress


@click.command()
@click.option("-d", "--db-string",
              type=click.Path(True, True, False, True, True, path_type=Path),
              required=True, help="Path to the database.")
@click.option("-s", "--study-id", type=int, required=True, help="Id of the study.")
@click.option("-p", "--progress-new",
              type=click.IntRange(min(AnnotationState).value, max(AnnotationState).value),
              required=True, default=0, help="Progress state for new patient entries with an existing label. " +
                                             ", ".join(f"{state.value}: {state.name}" for state in AnnotationState))
@click.option("-l", "--lost-folder",
              type=click.Path(False, False, True, readable=True, path_type=Path),
              required=True, help="Path to a lost-and-found folder where to move labels wit hno matching image.")
def main(db_string: Path, study_id: int, progress_new: int, lost_folder: Path) -> int:
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
        print(f"Study found (id={study_id}, name={study.name}):")
        db_patients = set(study.progress.keys())
    except NotFoundError:
        logger.error(f"Study {study_id} does not exist.")
        return 1

    # Now also look for files found on the hard drive
    img_folder = db_string.parent / study.img_folder
    lbl_folder = db_string.parent / study.label_folder
    images_found = {p.name for p in img_folder.iterdir()}
    labels_found = {p.name for p in lbl_folder.iterdir()}
    patients_all_good = images_found.intersection(labels_found)
    patients_only_img = images_found.difference(patients_all_good)
    patients_only_lbl = labels_found.difference(patients_all_good)

    # First handle the existing entries
    print(f"Found {len(db_patients.intersection(patients_all_good))} "
          f"entries in the database with both image and label.")

    db_entries_to_reinit = db_patients.intersection(patients_only_img)
    print(f"Found {len(db_entries_to_reinit)} "
          f"entries in the database with only an image (entries will be updated and labels will be created).")
    progress_update_to_study(
        study_id,
        {
            patient_name: AnnotationProgress(INIT_READER, AnnotationState.EMPTY, datetime.now(), "")
            for patient_name in db_entries_to_reinit
        }
    )
    # Create new labels
    for patient_name in db_entries_to_reinit:
        img = nib.load(img_folder / patient_name)
        empty_lbl = np.zeros(img.shape, dtype=np.uint8)
        nib.save(nib.Nifti1Image(empty_lbl, img.affine, img.header), lbl_folder / patient_name)

    entries_to_delete = db_patients.intersection(patients_only_lbl).union(db_patients.difference(images_found))
    print(f"Found {len(entries_to_delete)} "
          f"entries in the database no image or with only a label (entries will be deleted).")
    progress_delete_from_study(study_id, *list(entries_to_delete))

    # Then handle new images found on the disk
    new_images = images_found.difference(db_patients)
    new_images_with_labels = new_images.intersection(labels_found)
    new_images_no_labels = new_images.difference(labels_found)
    default_progress = AnnotationState(progress_new)

    print(f"Found {len(new_images_with_labels)} "
          f"images found on the hard drive (not in the database) with both image and label.")
    progress_add_to_study(
        study_id,
        {
            patient_name: AnnotationProgress(INIT_READER, default_progress, datetime.now(), "")
            for patient_name in new_images_with_labels
        }
    )

    print(f"Found {len(new_images_no_labels)} "
          f"found on the hard drive (not in the database) with no label.")
    progress_add_to_study(
        study_id,
        {
            patient_name: AnnotationProgress(INIT_READER, AnnotationState.EMPTY, datetime.now(), "")
            for patient_name in new_images_no_labels
        }
    )
    # Create new labels
    for patient_name in new_images_no_labels:
        img = nib.load(img_folder / patient_name)
        empty_lbl = np.zeros(img.shape, dtype=np.uint8)
        nib.save(nib.Nifti1Image(empty_lbl, img.affine, img.header), lbl_folder / patient_name)

    print(f"Found {len(db_patients.intersection(patients_only_lbl))} "
          f"files on the hard drive with only a label (files will be moved).")
    lost_folder.mkdir(exist_ok=True, parents=True)
    for patient_name in patients_only_lbl:
        (lbl_folder / patient_name).rename(lost_folder / patient_name)

    return 0


if __name__ == "__main__":
    exit(main())
