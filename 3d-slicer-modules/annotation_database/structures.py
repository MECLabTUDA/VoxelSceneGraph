from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum, auto
from pathlib import Path

STUDY_TABLE = "study"
PROGRESS_TABLE = "progress"


class AnnotationState(IntEnum):
    """State of the annotation for a given image."""
    EMPTY = 0
    AI_SEGMENTATION = auto()
    IN_PROGRESS = auto()
    MANUAL_SEGMENTATION = auto()
    PENDING_REVIEW_LVL1 = auto()
    PENDING_REVIEW_LVL2 = auto()
    REVIEWED = auto()

    def to_str(self) -> str:
        """String representation adapted to be displayed in a narrow table."""
        if self == AnnotationState.EMPTY:
            return "Empty"
        elif self == AnnotationState.AI_SEGMENTATION:
            return "AI Segmentation"
        elif self == AnnotationState.IN_PROGRESS:
            return "In Progress"
        elif self == AnnotationState.MANUAL_SEGMENTATION:
            return "Manual\nSegmentation"
        elif self == AnnotationState.PENDING_REVIEW_LVL1:
            return "Pending\nReview Lvl1"
        elif self == AnnotationState.PENDING_REVIEW_LVL2:
            return "Pending\nReview Lvl2"
        elif self == AnnotationState.REVIEWED:
            return "Reviewed"


@dataclass
class AnnotationProgress:
    """Progress for a given image."""
    _DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    reader: str
    progress: AnnotationState
    when: datetime
    comment: str

    def dict(self) -> dict:
        return {
            "reader": self.reader,
            "progress": self.progress.value,
            "when_": self.when.strftime(self._DATE_FORMAT),
            "comment": self.comment
        }

    @staticmethod
    def from_dict(data: dict) -> AnnotationProgress:
        return AnnotationProgress(
            data["reader"],
            AnnotationState(data["progress"]),
            datetime.strptime(data["when_"], AnnotationProgress._DATE_FORMAT),
            data["comment"]
        )


@dataclass
class Study:
    """A given study."""
    id: int | None  # Leave as None if not yet added to the database, will get updated by it
    name: str
    img_folder: Path
    label_folder: Path
    segments: list[str]
    last_segment_can_repeat: bool
    progress: dict[str, AnnotationProgress]
    window_width: int | None = None
    window_length: int | None = None

    def dict(self) -> dict:
        ret = {
            # "id": self.id,
            "name": self.name,
            "img_folder": self.img_folder.as_posix(),
            "label_folder": self.label_folder.as_posix(),
            "segments": ",".join(self.segments),
            "last_segment_can_repeat": self.last_segment_can_repeat,
            "window_width": self.window_width,
            "window_length": self.window_length
        }
        if self.id is not None:
            ret["id"] = self.id
        return ret

    @staticmethod
    def from_dict(data: dict) -> Study:
        return Study(
            data["id"],
            data["name"],
            Path(data["img_folder"]),
            Path(data["label_folder"]),
            data["segments"].split(","),
            data["last_segment_can_repeat"],
            # {k: AnnotationProgress.from_dict(prog) for k, prog in data["progress"].items()},
            {},
            data.get("window_width"),
            data.get("window_length")
        )
