import sqlite3
from pathlib import Path

import slicer

from scene_graph_database.functions import study_get_all
from scene_graph_database.options import Options
from scene_graph_database.structures import Study
from .load_state import LOAD_STATE_MANAGER


def load_studies_from_db(db_string: str, ignore_error: bool = False) -> bool:
    """Access the db, store config if it worked, load studies and populate widgets."""
    # Before we do anything, check that we do not have any unsaved changes to the patient
    if LOAD_STATE_MANAGER.selected_patient_scene_graph_modified and not confirm_discard_changes("Reloading the database"):
        return False

    if not Path(db_string).is_file():
        if not ignore_error:
            slicer.util.confirmOkCancelDisplay(
                f"Could not access database: path ({db_string}) does not exist or is not a file.",
                windowTitle="Database unreachable", parent=None
            )
        return False

    # Attempt to connect to the database
    Options.set_db_string(db_string)
    try:
        Options.get_db()
        print(f"Connected to: {db_string}")
    except sqlite3.OperationalError as e:
        if not ignore_error:
            slicer.util.confirmOkCancelDisplay(
                f"Could not access database: {e}",
                windowTitle="Database unreachable", parent=None
            )
        return False

    # Load studies and populate the combobox
    studies: list[Study] = list(study_get_all(fetch_progress=True).values())
    LOAD_STATE_MANAGER.study_list = studies
    return True


def confirm_discard_changes(action: str) -> bool:
    """Opens a dialog asking whether we want to discard changes to a modified patient."""
    return slicer.util.confirmOkCancelDisplay(
        "Some modifications to the current progress are not saved. "
        f"{action} will erase it. Do you want to continue?",
        windowTitle="Unsaved progress", parent=None
    )
