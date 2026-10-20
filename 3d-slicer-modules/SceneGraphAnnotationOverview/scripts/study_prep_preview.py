"""
Prepare preview images for a study. It's fine it you do it your way, but this script will help you.
Give the target resolution, "*" wildcards are accepted to keep the resolution along specific axes.
Note: Currently only available for 3D.
"""
from __future__ import annotations

import sqlite3
from logging import getLogger
from pathlib import Path

import click
import click_logging
import nibabel as nib
import numpy as np
import tqdm
from nibabel.processing import resample_from_to
from sqlite_utils.db import NotFoundError

from scene_graph_database.functions import study_get
from scene_graph_database.options import Options
from scene_graph_database.structures import StudyStructure


class StarOrFloat(click.ParamType):
    name = "star_or_float"

    def convert(self, value, param, ctx):
        # 1. Check for the special string
        if value == "*":
            return value

        # 2. Try to convert to float
        try:
            return float(value)
        except ValueError:
            self.fail(f"{value!r} is not '*' or a valid float", param, ctx)


@click.command()
@click.option("-d", "--db-string",
              type=click.Path(True, True, False, True, True, path_type=Path),
              required=True, help="Path to the database.")
@click.option("-s", "--study-id", type=int, required=True, help="Id of the study.")
@click.option("-r", "--resolution", type=StarOrFloat(), required=True, nargs=3,
              help="Target resolution for the preview images. Should be specified for each dimension. "
                   "Use \"*\" as wildcard for the dimension to keep its size as is. "
                   "Values are expected to be SPL orientation (depth-first for axial imaging).")
def main(db_string: Path, study_id: int, resolution: tuple) -> int:
    logger = getLogger(__file__)
    click_logging.basic_config(logger)

    # Validate the target resolution
    if all(map(lambda x: x == "*", resolution)):
        logger.error("You need to specify a resolution for at least one dimension. Otherwise, it's a no-op.")
        return 1
    if any(map(lambda x: x == 0., resolution)):
        logger.error("A resolution of 0 along any axis is not allowed. Please use a non-zero value.")
        return 1

    # Try to get access to the db to get the study location
    Options.set_db_string(db_string.as_posix())
    try:
        Options.get_db()
    except sqlite3.OperationalError as e:
        logger.error(f"Could not access database: {e}")
        return 1

    try:
        study = study_get(study_id, True)
        if study is None:
            logger.error(f"Study {study_id} does not exist.")
            return 1
    except NotFoundError:
        logger.error(f"Study {study_id} does not exist.")
        return 1

    # Mostly ignore patients in the database, we only process what we find.
    # But we still notify the user in case of discrepancy
    structure = StudyStructure(db_string, study.knowledge_graph_path)
    images_found = {p.name for p in structure.images.glob("*.nii.gz")}
    if len(images_found) != len(study.progress):
        logger.warning(
            f"Found {len(images_found)} images in the study, but {len(study.progress)} patients in the database. "
            f"You may need to update the progress for this study with the appropriate command."
        )

    # Process the images
    print(f"Preparing {len(images_found)} preview images for study (id={study_id}, name={study.name}).")
    target_orient = ("S", "P", "L")
    for image_name in tqdm.tqdm(sorted(images_found)):
        if (structure.images_preview / image_name).exists():
            # Already exists, skip
            continue

        # Load depth-first
        # noinspection PyUnresolvedReferences
        img = nib.load(structure.images / image_name).as_reoriented(np.array([[2, 1], [1, 1], [0, 1]]))
        orig_orientation = nib.aff2axcodes(img.affine)

        if orig_orientation != target_orient:
            # Reorient image if needed
            ornt_current = nib.io_orientation(img.affine)
            ornt_target = nib.orientations.axcodes2ornt(target_orient)
            transform = nib.orientations.ornt_transform(ornt_current, ornt_target)
            img = img.as_reoriented(transform)

        # Resample
        scales = np.array(tuple(1. if n == "*" else n / o for o, n in zip(img.header.get_zooms(), resolution)) + (1.,))
        new_aff = img.affine * scales
        new_shape = tuple(int(round(o / n)) for o, n in zip(img.shape, scales))
        img = resample_from_to(img, (new_shape, new_aff), mode="nearest")

        if orig_orientation != target_orient:
            # Reorient image back
            # noinspection PyUnboundLocalVariable
            transform = nib.orientations.ornt_transform(ornt_target, ornt_current)
            img = img.as_reoriented(transform)

        # Save the final image
        nib.save(img.as_reoriented(np.array([[2, 1], [1, 1], [0, 1]])), structure.images_preview / image_name)

    return 0


if __name__ == "__main__":
    exit(main())
