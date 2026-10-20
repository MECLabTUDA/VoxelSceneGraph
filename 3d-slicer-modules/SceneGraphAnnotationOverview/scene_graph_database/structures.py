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
    SEGMENTATION_IN_PROGRESS = auto()
    MANUAL_SEGMENTATION = auto()
    PENDING_REVIEW = auto()
    REVIEW_IN_PROGRESS = auto()
    REVIEWED = auto()

    def to_str(self) -> str:
        """String representation adapted to be displayed in a narrow table."""
        if self == AnnotationState.EMPTY:
            return "Empty"
        elif self == AnnotationState.AI_SEGMENTATION:
            return "AI Segmentation"
        elif self == AnnotationState.SEGMENTATION_IN_PROGRESS:
            return "Segmentation\nIn Progress"
        elif self == AnnotationState.MANUAL_SEGMENTATION:
            return "Manual\nSegmentation"
        elif self == AnnotationState.PENDING_REVIEW:
            return "Pending\nReview"
        elif self == AnnotationState.REVIEW_IN_PROGRESS:
            return "Review\nIn Progress"
        elif self == AnnotationState.REVIEWED:
            return "Reviewed"
        raise ValueError(f"Unknown annotation state: {self}")


@dataclass
class AnnotationProgress:
    """Progress for a given image."""
    _DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    reader: str
    progress: AnnotationState
    when: datetime

    def dict(self) -> dict:
        return {
            "reader": self.reader,
            "progress": self.progress.value,
            "when_": self.when.strftime(self._DATE_FORMAT),
        }

    @staticmethod
    def from_dict(data: dict) -> AnnotationProgress:
        return AnnotationProgress(
            data["reader"],
            AnnotationState(data["progress"]),
            datetime.strptime(data["when_"], AnnotationProgress._DATE_FORMAT),
        )


class StudyStructure:
    """
    Expected folder structure for the annotation process.
    All paths are relative to the Knowledge Graph path for a study.
    """
    # Folder for the (HR) images
    _images: Path = Path("./images")
    # Folder for the annotated Scene Graphs
    _scene_graphs: Path = Path("./psgg")

    # (Optional) Folder for any masks/AI predictions that we may want to inspect
    # Note: we can also automatically generate Scene Graphs from these
    _ai_predictions: Path = Path("./pred")
    # (Optional) Folder for preview (LR) images. HR images may be slow to load at full resolution.
    _images_preview: Path = Path("./images_preview")
    # (Optional) Folder for the PDF reports
    _reports: Path = Path("./reports")

    def __init__(self, db_string: Path, knowledge_graph_path: Path):
        self.knowledge_graph = (db_string.parent / knowledge_graph_path).resolve()
        self.root = self.knowledge_graph.parent

    @property
    def ai_predictions(self) -> Path:
        res = self.root / self._ai_predictions
        if not res.exists():
            res.mkdir(parents=True, exist_ok=True)
        return res

    @property
    def images(self) -> Path:
        res = self.root / self._images
        if not res.exists():
            res.mkdir(parents=True, exist_ok=True)
        return res

    @property
    def images_preview(self) -> Path:
        res = self.root / self._images_preview
        if not res.exists():
            res.mkdir(parents=True, exist_ok=True)
        return res

    @property
    def scene_graphs(self) -> Path:
        res = self.root / self._scene_graphs
        if not res.exists():
            res.mkdir(parents=True, exist_ok=True)
        return res

    @property
    def reports(self) -> Path:
        res = self.root / self._reports
        if not res.exists():
            res.mkdir(parents=True, exist_ok=True)
        return res


@dataclass
class Study:
    """A given study."""
    id: int | None  # Leave as None if not yet added to the database, will get updated by it
    name: str
    knowledge_graph_path: Path
    progress: dict[str, AnnotationProgress]

    def dict(self) -> dict:
        ret = {
            # "id": self.id,
            "name": self.name,
            "knowledge_graph_path": self.knowledge_graph_path.as_posix(),
        }
        if self.id is not None:
            # noinspection PyTypeChecker
            ret["id"] = self.id
        return ret

    @staticmethod
    def from_dict(data: dict) -> Study:
        return Study(
            data["id"],
            data["name"],
            knowledge_graph_path=Path(data["knowledge_graph_path"]),
            progress={},
        )
