"""
Copyright 2023 Antoine Sanner, Technical University of Darmstadt, Darmstadt, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging

from PyQt6.QtWidgets import QMessageBox, QWidget
from scene_graph_api.logging_handlers import RecordCountingHandler


class RecordDisplayHandler(RecordCountingHandler):
    """Used to display a QMessageBox if warning/errors came up during processing."""

    def display_records(self, parent: QWidget):
        """Opens a QMessageBox if warning/errors are registered."""
        msg = ""
        if self._records_seen.get(logging.ERROR):
            msg += "Error(s) found:\n\t-" + \
                   f"\n\t-".join([rec.msg for rec in self._records_seen.get(logging.ERROR, [])])

        if self._records_seen.get(logging.WARN):
            msg += "\nWarning(s) found:\n\t-" + \
                   f"\n\t-".join([rec.msg for rec in self._records_seen.get(logging.WARNING, [])])

        if msg:
            QMessageBox.warning(parent, "Warning", msg)

    def has_warnings(self) -> bool:
        """Returns whether records of level logging.WARN are stored."""
        return bool(self._records_seen.get(logging.WARN, []))

    def has_errors(self) -> bool:
        """Returns whether records of level logging.ERROR are stored."""
        return bool(self._records_seen.get(logging.ERROR, []))


class TestingHandler(RecordDisplayHandler):
    """Should only be used in tests to check that issues have been found (esp. when parsing JSON files)."""

    def __init__(self):
        super().__init__()
        self.display_records_called = False

    def get_warning_message_count(self) -> int:
        """Returns the number of records of level logging.WARN stored."""
        return 0 if logging.WARN not in self._records_seen else len(self._records_seen[logging.WARN])

    def get_error_message_count(self):
        """Returns the number of records of level logging.ERROR stored."""
        return 0 if logging.ERROR not in self._records_seen else len(self._records_seen[logging.ERROR])

    def print_messages(self):
        """Prints the message of all records stored. Should only be used when debugging failing tests."""
        for k in self._records_seen:
            print(f"{k}:")
            for record in self._records_seen[k]:
                print(f"\t{record.msg}")

    def display_records(self, parent: QWidget):
        """Instead of displaying anything we flag that the method was called for testing purposes."""
        self.display_records_called = True

    def purge(self):
        super().purge()
        self.display_records_called = False
