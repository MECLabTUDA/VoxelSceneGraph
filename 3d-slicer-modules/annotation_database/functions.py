"""List of functions that can be used elsewhere for the most common purposes."""
from __future__ import annotations

import sqlite3
from typing import Iterable

from .options import Options
from .structures import Study, AnnotationState, AnnotationProgress, STUDY_TABLE, PROGRESS_TABLE

INIT_READER = "Init"


# ======================================================================================================================
# Tables init/delete
# ======================================================================================================================
def init_tables():
    """Initialize the tables."""
    db = Options.get_db()
    db.executescript(f"""
        CREATE TABLE IF NOT EXISTS {STUDY_TABLE} (
            id integer PRIMARY KEY,
            name text NOT NULL,
            img_folder text NOT NULL,
            label_folder text NOT NULL,
            segments text NOT NULL,
            last_segment_can_repeat boolean NOT NULL,
            window_width integer,
            window_length integer
                CHECK((window_width IS NULL and window_length IS NULL) OR 
                      (window_width IS NOT NULL and window_length IS NOT NULL))
        );
        CREATE TABLE IF NOT EXISTS {PROGRESS_TABLE} (
            study_id integer,
            patient_name text NOT NULL,
            reader text NOT NULL,
            progress integer NOT NULL
                DEFAULT 0
                CHECK(progress >= {min(AnnotationState).value})
                CHECK(progress <= {max(AnnotationState).value}),
            when_ text NOT NULL,
            comment text NOT NULL,
            FOREIGN KEY (study_id) REFERENCES {STUDY_TABLE} (id),
            PRIMARY KEY (study_id, patient_name)
        );
        """)


def delete_tables():
    """Deletes the tables."""
    db = Options.get_db()
    try:
        db[PROGRESS_TABLE].drop()
    except sqlite3.OperationalError:
        pass
    try:
        db[STUDY_TABLE].drop()
    except sqlite3.OperationalError:
        pass


# ======================================================================================================================
# Study Progress
# ======================================================================================================================
def progress_get_for_study(study_id: int) -> dict[str, AnnotationProgress]:
    """Add records where study_id=study_id."""
    db = Options.get_db()
    return {
        row["patient_name"]: AnnotationProgress.from_dict(row)
        for row in db[PROGRESS_TABLE].rows_where("study_id = ?", [study_id])
    }


def progress_add_to_study(study_id: int, progresses: dict[str, AnnotationProgress]):
    """Add records where study_id=study_id."""
    if not progresses:
        return

    db = Options.get_db()
    # upsert_all somehow not working...
    db[PROGRESS_TABLE].insert_all([
        {"study_id": study_id, "patient_name": patient_name, **progress.dict()}
        for patient_name, progress in progresses.items()
    ], pk=("study_id", "patient_name"))


def progress_update_to_study(study_id: int, progresses: dict[str, AnnotationProgress]):
    """Add records where study_id=study_id."""
    if not progresses:
        return

    db = Options.get_db()
    # upsert_all somehow not working...
    db[PROGRESS_TABLE].upsert_all([
        {"study_id": study_id, "patient_name": patient_name, **progress.dict()}
        for patient_name, progress in progresses.items()
    ], pk=("study_id", "patient_name"))


def progress_delete_all_from_studies(*study_ids: int):
    """Delete all records where study_id=study_id."""
    if len(study_ids) == 0:
        return
    elif len(study_ids) == 1:
        op_str = f"= {study_ids[0]}"
    else:
        op_str = f"IN {study_ids}"

    db = Options.get_db()
    db[PROGRESS_TABLE].delete_where(f"study_id {op_str}")


def progress_delete_from_study(study_id: int, *patient_names: str):
    """Delete all records for the given patient names where study_id=study_id."""
    if len(patient_names) == 0:
        return
    elif len(patient_names) == 1:
        op_str = f"= '{patient_names[0]}'"
    else:
        op_str = f"IN {patient_names}"

    db = Options.get_db()
    db[PROGRESS_TABLE].delete_where(f"study_id = ? AND patient_name {op_str}", [study_id])
    db.conn.commit()


def progress_update_from_patients_found(study_id: int, patient_names: Iterable[str]):
    """
    Given the study_id and a list of patient names (e.g. found on the hard drive):
    - Delete existing rows for names that are not in the list
    - Keep the state of existing rows for names that match
    - Add rows for new names that do not have any match
    """
    input_names = set(patient_names)
    db_names = set(progress_get_for_study(study_id).keys())
    progress_delete_from_study(study_id, *list(db_names.difference(input_names)))
    progress_add_to_study(
        study_id,
        {name: AnnotationProgress(INIT_READER, AnnotationState.EMPTY, "") for name in input_names.difference(db_names)}
    )


# ======================================================================================================================
# Studies
# ======================================================================================================================
def study_get_all(fetch_progress: bool = False) -> dict[int, Study]:
    """Return all studies."""
    db = Options.get_db()
    ret = {row["id"]: Study.from_dict(row) for row in db[STUDY_TABLE].rows}

    if fetch_progress:
        for study in ret.values():
            study.progress = progress_get_for_study(study.id)

    return ret


def study_get(study_id: int, fetch_progress: bool = False) -> Study | None:
    """
    Return a given study.
    :raise sqlite_utils.db.NotFoundError: if not found
    """
    db = Options.get_db()
    study = Study.from_dict(db[STUDY_TABLE].get(study_id))
    if fetch_progress:
        study.progress = progress_get_for_study(study_id)
    return study


def study_add(study: Study, patient_names: list[str] | None = None):
    """Add a study to the database."""
    db = Options.get_db()
    table = db[STUDY_TABLE]
    table.insert(study.dict())
    study.id = table.last_pk

    if patient_names:
        progress_add_to_study(
            study.id,
            {name: AnnotationProgress(INIT_READER, AnnotationState.EMPTY, "") for name in patient_names}
        )


def study_id_delete(*study_ids: int):
    """Delete all studies specified."""
    if not study_ids:
        return

    db = Options.get_db()
    progress_delete_all_from_studies(*study_ids)

    if len(study_ids) == 1:
        op_str = f"= {study_ids[0]}"
    else:
        op_str = f"IN {tuple(study_id for study_id in study_ids)}"

    db[STUDY_TABLE].delete_where(f"id {op_str}")
    db.conn.commit()


def study_delete(*studies: Study):
    """Delete all studies specified."""
    if not studies:
        return

    db = Options.get_db()
    progress_delete_all_from_studies(*[study.id for study in studies])

    if len(studies) == 1:
        op_str = f"= {studies[0].id}"
    else:
        op_str = f"IN {tuple(study.id for study in studies)}"

    db[STUDY_TABLE].delete_where(f"id {op_str}")
