"""
Python script to create a new study and/or to update the annotation progress.
It is equivalent to running these 3 scripts in sequence:
- database_rebuild.py
- study_add.py
- study_progress_update.py
"""

import sys
from logging import getLogger, StreamHandler
from pathlib import Path

from scene_graph_database.scripts import study_add, study_progress_update, database_rebuild
from scene_graph_database.structures import AnnotationState

# Set the database path (will be created, if not existent)
# E.g. "/media/user/Creator Pro/SGG/annotation.db"
db_path = Path("path to the database file")
# Set the knowledge graph path. Either needs to be relative to the database, or an absolute path in a subfolder.
# E.g. "/media/user/Creator Pro/SGG/My Study/nowledge_graph.json"
knowledge_graph_path = Path("path to the knowledge graph file")

# Set up either study id or name for a new study
# If the id is 0, a new study will be created
# If not 0, the script will just update the annotation progress for new/existing cases
study_id = 0
study_name = "Localizer"

# ======================================================================================================================
# Setting up the logger
logger = getLogger(__name__)
logger.setLevel("INFO")
ch = StreamHandler(sys.stdout)
ch.setLevel("INFO")
logger.addHandler(ch)

# Regenerating the database if needed
if not db_path.exists():
    database_rebuild(db_path, logger)

if study_id == 0:
    # Create a new study
    study_id, ret = study_add(db_path, study_name, knowledge_graph_path, logger)
    if ret != 0:
        logger.info("Could not add study")
        exit(ret)

# Update progress
ret = study_progress_update(db_path, study_id, AnnotationState.AI_SEGMENTATION.value, Path("."), logger)
if ret != 0:
    logger.info("Could not update progress")
