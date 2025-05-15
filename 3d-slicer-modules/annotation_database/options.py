"""Options for the database."""
from __future__ import annotations

from dataclasses import dataclass

from sqlite_utils import Database
import sqlite3


@dataclass
class Options:
    # Path to the local database
    _DB_STRING = "test.db"
    _db: Database | None = None

    @staticmethod
    def get_db_string() -> str:
        return Options._DB_STRING

    @staticmethod
    def set_db_string(db_string: str):
        Options._DB_STRING = db_string
        Options._db = None

    @staticmethod
    def get_db() -> Database:
        if Options._db is None:
            Options._db = Database(Options._DB_STRING)
        try:
            # Try to get table names as a check for broken connections
            Options._db.view_names()
        except sqlite3.OperationalError:
            # Try to renew the connection
            Options._db = Database(Options._DB_STRING)
        return Options._db
