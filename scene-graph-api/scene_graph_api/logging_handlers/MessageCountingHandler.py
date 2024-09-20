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
from logging import Handler, LogRecord


class RecordCountingHandler(Handler):
    """Logging handler used for counting warnings and errors."""

    def __init__(self):
        super().__init__()
        self._records_seen: dict[int, list[LogRecord]] = {}

    def emit(self, record: LogRecord):
        """Adds the new record to the dict of records seen based on the log level."""
        if record.levelno not in self._records_seen:
            self._records_seen[record.levelno] = []
        self._records_seen[record.levelno].append(record)

    def purge(self):
        """Clears the dict of records stored."""
        self._records_seen = {}

    def has_warnings(self) -> bool:
        """Returns whether records of level logging.WARN are stored."""
        return bool(self._records_seen.get(logging.WARN, []))

    def has_errors(self) -> bool:
        """Returns whether records of level logging.ERROR are stored."""
        return bool(self._records_seen.get(logging.ERROR, []))


class TestingHandler(RecordCountingHandler):
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

    def purge(self):
        super().purge()
        self.display_records_called = False
