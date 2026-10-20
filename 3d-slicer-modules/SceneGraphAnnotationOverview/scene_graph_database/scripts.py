from __future__ import annotations

import sqlite3
from datetime import datetime
from logging import Logger, getLogger
from pathlib import Path

import numpy as np
import tqdm
from sqlite_utils.db import NotFoundError

from scene_graph_api.knowledge import KnowledgeGraph
from scene_graph_api.scene import SceneGraph
from scene_graph_api.utils.nifti_io import NiftiImageWrapper
from scene_graph_api.utils.pathing import remove_suffixes
from scene_graph_database.functions import study_add as sql_study_add, study_id_delete, study_get_all, study_get, \
    progress_delete_from_study, progress_update_to_study, progress_add_to_study, delete_tables, init_tables
from scene_graph_database.options import Options
from scene_graph_database.structures import AnnotationState, AnnotationProgress, StudyStructure, Study


def database_rebuild(db_string: Path, logger: Logger | None = None) -> int:
    """Rebuild the database and delete all existing content or create an entirely new one if it does not exist."""
    if logger is None:
        logger = getLogger(__file__)

    # First try to get access to the db
    Options.set_db_string(db_string.as_posix())
    try:
        Options.get_db()
    except sqlite3.OperationalError as e:
        logger.error(f"Could not access database: {e}")
        return 1

    delete_tables()
    init_tables()

    logger.info("Database rebuilt.")
    return 0


def study_add(
        db_string: Path,
        name: str,
        knowledge_graph_path: Path,
        logger: Logger | None = None
) -> tuple[int, int]:
    """Add a new study to the database."""
    if logger is None:
        logger = getLogger(__file__)

    # Catch absolute paths for the knowledge graph
    if knowledge_graph_path.is_absolute():
        if not knowledge_graph_path.is_relative_to(db_string.parent):
            logger.error(f"Knowledge graph {knowledge_graph_path} is whether a relative path nor "
                         f"an absolute path relative to the database.")
            return 0, 1
        knowledge_graph_path = knowledge_graph_path.relative_to(db_string.parent)

    # Check that the knowledge graph exists
    abs_knowledge_graph_path = (db_string.parent / knowledge_graph_path).resolve()
    if not abs_knowledge_graph_path.exists():
        logger.error(f"Knowledge graph {abs_knowledge_graph_path} does not exist.")
        return 0, 1

    # First try to get access to the db
    Options.set_db_string(db_string.as_posix())
    try:
        Options.get_db()
    except sqlite3.OperationalError as e:
        logger.error(f"Could not access database: {e}")
        return 0, 1

    # Check that the knowledge graph is valid
    if logger is not None:
        logger.setLevel(0)
    kg = KnowledgeGraph.load(abs_knowledge_graph_path, logger)
    if kg is None:
        logger.warning(f"Knowledge graph {abs_knowledge_graph_path} is not valid.")

    # Build the study object
    study = Study(
        id=None,  # Auto filled
        name=name,
        knowledge_graph_path=knowledge_graph_path,
        progress={}
    )
    sql_study_add(study)

    logger.info(f"Added the study (id={study.id}).")
    return study.id, 0


def study_delete(db_string: Path, study_id: list[int], logger: Logger | None = None) -> int:
    """Delete a study from the database."""
    if logger is None:
        logger = getLogger(__file__)

    # First try to get access to the db
    Options.set_db_string(db_string.as_posix())
    try:
        Options.get_db()
    except sqlite3.OperationalError as e:
        logger.error(f"Could not access database: {e}")
        return 1

    study_id_delete(*study_id)
    logger.info("Deleted studies with matching ids. Double check using the study listing command.")
    return 0


def study_list(db_string: Path, logger: Logger | None = None) -> int:
    """List all studies."""
    if logger is None:
        logger = getLogger(__file__)

    # First try to get access to the db
    Options.set_db_string(db_string.as_posix())
    try:
        Options.get_db()
    except sqlite3.OperationalError as e:
        logger.error(f"Could not access database: {e}")
        return 1

    studies = study_get_all(False)
    logger.info(f"{len(studies)} stud{'y' if len(studies) < 2 else 'ies'} found:")
    for study in studies.values():
        logger.info(f"\t{study}")

    return 0


def study_progress_list(db_string: Path, study_id: int, logger: Logger | None = None) -> int:
    """List all files and their progress for a given study."""
    if logger is None:
        logger = getLogger(__file__)

    # First try to get access to the db
    Options.set_db_string(db_string.as_posix())
    try:
        Options.get_db()
    except sqlite3.OperationalError as e:
        logger.error(f"Could not access database: {e}")
        return 1

    try:
        study = study_get(study_id, True)
        logger.info(f"Found {len(study.progress)} files for study (id={study_id}, name={study.name}):")
        for patient_name, progress in study.progress.items():
            logger.info(f"\t{patient_name}: {progress}")

    except NotFoundError:
        logger.error(f"Study {study_id} does not exist.")
        return 1

    return 0


def study_progress_update(
        db_string: Path,
        study_id: int,
        progress_new: int,
        lost_folder: Path,
        logger: Logger | None = None
) -> int:
    """Update the progress state for a given study."""
    if logger is None:
        logger = getLogger(__file__)

    # First try to get access to the db
    Options.set_db_string(db_string.as_posix())
    try:
        Options.get_db()
    except sqlite3.OperationalError as e:
        logger.error(f"Could not access database: {e}")
        return 1

    try:
        study = study_get(study_id, True)
        logger.info(f"Study found (id={study_id}, name={study.name}):")
        db_patients = set(study.progress.keys())
    except NotFoundError:
        logger.error(f"Study {study_id} does not exist.")
        return 1

    structure = StudyStructure(db_string, study.knowledge_graph_path)

    # Load the knowledge graph
    if not structure.knowledge_graph.exists():
        logger.error(f"Knowledge graph {structure.knowledge_graph} does not exist.")
        return 1

    kg = KnowledgeGraph.load(structure.knowledge_graph, logger)
    if kg is None:
        logger.error(f"Knowledge graph {structure.knowledge_graph} is not valid.")
        return 1

    # Now also look for files found on the hard drive
    images_found = {p.name for p in structure.images.glob("*.nii.gz")}
    # Replace JSON file ext for matching
    labels_found = {remove_suffixes(p) + ".nii.gz" for p in structure.scene_graphs.glob("*.json")}
    preds_found = {p.name for p in structure.ai_predictions.glob("*.nii.gz")}
    # Patients with both image and SG
    patients_all_good = images_found.intersection(labels_found)
    # Patients with both image and SG already in the database: nothing to do
    patients_all_good_existing = db_patients.intersection(patients_all_good)
    # Note: we don't care whether an image without a SG is already in the DB
    patients_only_img = images_found.difference(patients_all_good)
    patients_only_img_with_pred = patients_only_img.intersection(preds_found)
    patients_only_img_no_pred = patients_only_img.difference(patients_only_img_with_pred)
    patients_only_lbl = labels_found.difference(patients_all_good)

    logger.info(f"Found {len(patients_all_good_existing)} entries in the database with both image and Scene Graph.")

    # Add the easy cases that are not yet in the database
    patients_all_good_not_existing = patients_all_good.difference(db_patients)
    logger.info(
        f"Found {len(patients_all_good_not_existing)} patients with both image and Scene Graph, but not yet in the database.")
    progress_add_to_study(
        study_id,
        {patient_name: AnnotationProgress("Init", AnnotationState.EMPTY, datetime.now())
         for patient_name in patients_all_good_not_existing}
    )

    # Check for any entries in the database that don't have images
    # If we remove them, we can forget about existing entries and only focus on what we have found on the drive
    entries_to_delete = db_patients.difference(images_found)
    logger.info(f"Found {len(entries_to_delete)} "
          f"entries in the database without an image (entries will be deleted).")
    progress_delete_from_study(study_id, *list(entries_to_delete))

    # Then handle Scene Graphs that don't have images
    logger.info(f"Found {len(patients_only_lbl)} "
          f"files on the hard drive with only a label (files will be moved).")
    if patients_only_lbl:
        lost_folder.mkdir(exist_ok=True, parents=True)
        for patient_name in patients_only_lbl:
            (structure.scene_graphs / patient_name).rename(lost_folder / patient_name)

    # From nothing
    logger.info(f"Found {len(patients_only_img_no_pred)} "
          f"patients with only an image. "
          f"Empty Scene Graphs will be created.")
    errored = 0
    for patient_name in patients_only_img_no_pred:
        img = NiftiImageWrapper.load_depth_first(structure.images / patient_name)
        empty_seg = NiftiImageWrapper.from_array(np.zeros(img.shape, dtype=np.uint8), img.affine, img.header)
        sg = SceneGraph.create_from_segmentation(kg, empty_seg, logger)
        if sg is None:
            logger.error(f"Could not create Scene Graph for {patient_name}.")
            errored += 1
        else:
            sg.save(structure.scene_graphs / patient_name.replace(".nii.gz", ".json"))
            if patient_name in db_patients:
                progress_update_to_study(
                    study_id,
                    {patient_name: AnnotationProgress("Init", AnnotationState.AI_SEGMENTATION, datetime.now())}
                )
            else:
                progress_add_to_study(
                    study_id,
                    {patient_name: AnnotationProgress("Init", AnnotationState.AI_SEGMENTATION, datetime.now())}
                )

    # From AI prediction
    logger.info(f"Found {len(patients_only_img_with_pred)} "
          f"patients with only an image. "
          f"Scene Graphs will be created from available AI predictions.")
    errored = 0
    if patients_only_img_with_pred:
        for patient_name in tqdm.tqdm(patients_only_img_with_pred):
            prediction = NiftiImageWrapper.load_depth_first(structure.ai_predictions / patient_name)
            sg = SceneGraph.create_from_segmentation(kg, prediction, logger)
            if sg is None:
                logger.error(f"Could not create Scene Graph for {patient_name}.")
                errored += 1
            else:
                sg.save(structure.scene_graphs / patient_name.replace(".nii.gz", ".json"))
                if patient_name in db_patients:
                    progress_update_to_study(
                        study_id,
                        {patient_name: AnnotationProgress("AI", AnnotationState.AI_SEGMENTATION, datetime.now())}
                    )
                else:
                    progress_add_to_study(
                        study_id,
                        {patient_name: AnnotationProgress("AI", AnnotationState.AI_SEGMENTATION, datetime.now())}
                    )
    if errored > 0:
        logger.warning(f"Could not create Scene Graphs from AI predictions for {errored} patients.")

    return 0
