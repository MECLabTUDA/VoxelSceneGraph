import json
from pathlib import Path

import qt
import slicer

from scene_graph_api.utils.pathing import get_datadir
from .collapsible_button_widget import CollapsibleButtonWidget
from ..load_state import LOAD_STATE_MANAGER
from ..utils import load_studies_from_db


class ConfigWidget(CollapsibleButtonWidget):
    """Widget with parameters for the annotation overview. I.e. the path to the database and the reader name."""

    def __init__(self):
        super().__init__()

        # Where to find any saved parameters
        self._config_save_path = get_datadir() / "slicer_annotation_overview.json"

        # Relevant form elements
        self._db_string_text = qt.QLineEdit()
        self._reader_text = qt.QLineEdit()
        self._validate_button = qt.QPushButton("Load studies")

        self.init_ui()

    def init_ui(self):
        self.setText("General Configuration")

        layout = qt.QFormLayout()
        self.setButtonLayout(layout)
        layout.addRow(qt.QLabel("Database path:"), self._db_string_text)
        layout.addRow(qt.QLabel("Reader:"), self._reader_text)

        def f():
            LOAD_STATE_MANAGER.reader = self._reader_text.text

        self._reader_text.textChanged.connect(f)

        layout.addRow(self._validate_button)
        self._validate_button.clicked.connect(self.load_studies)

        # Try to load global config from disk
        if self._config_save_path.exists():
            try:
                with self._config_save_path.open("r") as f:
                    options = json.load(f)
                self._db_string_text.setText(options.get("db_string", ""))
                self._reader_text.setText(options.get("reader", ""))
                LOAD_STATE_MANAGER.reader = self._reader_text.text
                # Auto-load studies
                self.load_studies(ignore_error=True)
            except json.JSONDecodeError:
                ...

    def load_studies(self, ignore_error: bool = False) -> None:
        """Access the db, store config if it worked, load studies and populate widgets."""
        db_path = self._db_string_text.text
        if not Path(db_path).is_file():
            slicer.util.confirmOkCancelDisplay(
                "Database file does not exist. Please enter a valid path to the database file.",
                windowTitle="Database File Not Found", parent=None
            )
            return

        success = load_studies_from_db(db_path, ignore_error)

        if success:
            LOAD_STATE_MANAGER.reader = self._reader_text.text
            # Save options
            with open(self._config_save_path, "w") as f:
                json.dump({
                    "db_string": db_path,
                    "reader": self._reader_text.text
                }, f)
