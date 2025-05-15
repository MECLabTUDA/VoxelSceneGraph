import json
from collections import defaultdict
from enum import IntEnum, auto
from pathlib import Path


class Progress(IntEnum):
    INITIALIZED = auto()
    PENDING_REVIEW = auto()
    FINISHED = auto()


class CohortAnnotationProgress(defaultdict):
    """Default dict to store the annotation progress for each patient in a cohort."""
    FILE = "cohort_progress"

    def __init__(self, data: dict, scene_graph_folder: Path):
        super().__init__(lambda: Progress.INITIALIZED, **data)
        self._scene_graph_folder = scene_graph_folder
        self._path = scene_graph_folder / self.FILE

    def save(self):
        with open(self._path, "w") as f:
            json.dump({k: v.value for k, v in self.items()}, f)

    @classmethod
    def load(cls, scene_graph_folder: Path):
        """Quietly ignores fucked up data as it is not very important."""
        max_v = max(Progress).value
        try:
            with open(scene_graph_folder / cls.FILE, "r") as f:
                raw = json.load(f)
        except (json.decoder.JSONDecodeError, UnicodeDecodeError, FileNotFoundError):
            raw = {}

        if not isinstance(raw, dict):
            return cls({}, scene_graph_folder)

        return cls({
            k: (Progress(v) if isinstance(v, int) and 0 < v <= max_v else Progress.INITIALIZED)
            for k, v in raw.items()
        }, scene_graph_folder)
