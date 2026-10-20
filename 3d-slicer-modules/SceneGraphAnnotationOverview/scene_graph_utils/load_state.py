from PythonQt.QtCore import QObject
import qt

from scene_graph_api.knowledge import KnowledgeGraph
from scene_graph_api.scene import BoundingBox, Relation, SceneGraph
from scene_graph_database.structures import Study, AnnotationState


class LoadStateManager(QObject):
    """
    Class used to centralize all signals, options, ... related to the UI and loading study/patient data.
    """
    # ==================================================================================================================
    # Study/patient selection stuff
    # ==================================================================================================================
    # Signals that the study list was updated
    study_list_updated = qt.Signal()
    # Signals that a new study was selected
    study_selected = qt.Signal()
    # Signals that a new patient was selected
    patient_selected = qt.Signal()

    # I'm lazy, so I want to have access to the table easily for key binds from anywhere
    select_patient_by_offset = qt.Signal(int)

    # Signals that Scene Graph has been loaded
    sg_loaded = qt.Signal()

    # Signals that we should show the preview image of the selected patient
    preview_toggled = qt.Signal()
    # Signals that we should show the AI prediction of for selected patient
    ai_prediction_toggled = qt.Signal()

    # Used to tell the ParameterNodeHandler to remove a segment given an object id.
    # This in turn triggers all relevant stuff, including a "safe_remove_object_from_scene_graph" call
    remove_segment_for_object = qt.Signal(int)

    # ==================================================================================================================
    # Scene Graph stuff
    # ==================================================================================================================
    # Signals that the user selected another subject
    subject_changed = qt.Signal()
    # Signals that the user selected another object
    object_changed = qt.Signal()

    # Signals that the attributes name has changed
    # We cannot have a signal per object as signals need to be class attributes
    # So we pass the object as argument and let callees figure out if they need to do something
    name_changed = qt.Signal(BoundingBox)

    # If any segment was modified, notify the UI that the color might have changed
    update_color = qt.Signal()

    # Same but with a Bounding Box or Segmentation creation
    sg_object_added = qt.Signal(BoundingBox)
    # Same but with a Bounding Box or Segmentation deletion
    sg_object_removed = qt.Signal(BoundingBox)

    # Same but with a relation creation
    relation_added = qt.Signal(Relation)
    # Same but with a relation deletion
    relation_removed = qt.Signal(Relation)

    # Signals used to select the view to show the object with the given object to the user
    show_object = qt.Signal(BoundingBox)

    def __init__(self):
        super().__init__()

        # Study/patient selection stuff
        self._study_list: list[Study] | None = None
        self._selected_study_idx: int | None = None
        self._selected_patient_name: str | None = None
        # Anything stored in the SG
        self.selected_patient_scene_graph_modified: bool = False
        # Stuff saved in the annotation database (progress, comments, ...)
        # self.selected_patient_data_modified: bool = False

        self.colors: dict[int, str] = {}  # Object id to hex color

        # Also keep track of written comments and progress
        # Note: doing it here is easier than propagating the information through the widget hierarchy
        self.reader: str = ""
        self.annotation_progress: AnnotationState | None = None

        self._show_preview: bool = False
        self._show_ai_prediction: bool = False

        # Scene Graph stuff
        self.knowledge_graph: KnowledgeGraph | None = None
        self._scene_graph: SceneGraph | None = None
        self._subject: BoundingBox | None = None
        self._object: BoundingBox | None = None

    # ==================================================================================================================
    # Study/patient selection stuff
    # ==================================================================================================================
    @property
    def study_list(self) -> list[Study] | None:
        return self._study_list

    @study_list.setter
    def study_list(self, new_study_list: list[Study] | None):
        """Only emits if the new study list is different from the previous one."""
        old_study_list = self._study_list
        self._study_list = new_study_list
        if new_study_list != old_study_list:
            self._selected_study_idx = None
            self.knowledge_graph = None
            # Cascade reset
            self.selected_patient_name = None
            self.study_list_updated.emit()

    @property
    def selected_study_idx(self) -> int | None:
        return self._selected_study_idx

    @property
    def selected_study(self) -> Study | None:
        if self._study_list is None or self._selected_study_idx is None:
            return None
        return self._study_list[self._selected_study_idx]

    # @selected_study_idx.setter
    # def selected_study_idx(self, new_selected_study_idx: int | None):
    #     """Only emits if the new selected study index is different from the previous one."""
    #     old_selected_study_idx = self._selected_study_idx
    #     self._selected_study_idx = new_selected_study_idx
    #     if new_selected_study_idx != old_selected_study_idx:
    #         self.study_selected.emit()
    #         self.knowledge_graph = None
    #         # Cascade reset
    #         self.selected_patient_name = None

    def clear_study_selection(self):
        """Clears the study selection and emits the study_selected signal."""
        self._selected_study_idx = None
        self.knowledge_graph = None
        self.scene_graph = None
        self.study_selected.emit()
        # Cascade reset
        self.selected_patient_name = None

    def set_new_study_idx(self, idx: int, knowledge_graph: KnowledgeGraph):
        """
        Set both the study idx and knowledge graph both at once.
        Only emits if the new selected study index is different from the previous one.
        """
        old_selected_study_idx = self._selected_study_idx
        self._selected_study_idx = idx
        if idx != old_selected_study_idx:
            self.knowledge_graph = knowledge_graph
            self.study_selected.emit()
            # Cascade reset
            self.selected_patient_name = None

    @property
    def selected_patient_name(self) -> str | None:
        return self._selected_patient_name

    @selected_patient_name.setter
    def selected_patient_name(self, new_selected_patient_name: str | None):
        """Only emits if the new selected patient name is different from the previous one."""
        old_selected_patient_name = self._selected_patient_name
        self._selected_patient_name = new_selected_patient_name
        if new_selected_patient_name != old_selected_patient_name:
            # Init the progress member here
            if new_selected_patient_name is None:
                self.annotation_progress = None
            else:
                self.annotation_progress = self.selected_study.progress[self._selected_patient_name].progress

            self.patient_selected.emit()
            self.selected_patient_scene_graph_modified = False
            # Cascade reset
            # FIXME somehow the scene graph is already set for that patient
            # self.scene_graph = None

    @property
    def show_preview(self):
        return self._show_preview

    @show_preview.setter
    def show_preview(self, show_preview: bool):
        old_show_preview = self._show_preview
        self._show_preview = show_preview
        if self._show_preview != old_show_preview:
            self.preview_toggled.emit()

    @property
    def show_ai_prediction(self):
        return self._show_ai_prediction

    @show_ai_prediction.setter
    def show_ai_prediction(self, show_ai_prediction: bool):
        old_show_ai_prediction = self._show_ai_prediction
        self._show_ai_prediction = show_ai_prediction
        if self._show_ai_prediction != old_show_ai_prediction:
            self.ai_prediction_toggled.emit()

    # ==================================================================================================================
    # Scene Graph stuff
    # ==================================================================================================================
    @property
    def scene_graph(self) -> SceneGraph | None:
        return self._scene_graph

    @scene_graph.setter
    def scene_graph(self, new_scene_graph: SceneGraph | None):
        """Only emits if the new scene graph is different from the previous one."""
        old_scene_graph = self._scene_graph
        self._scene_graph = new_scene_graph

        if new_scene_graph != old_scene_graph:
            self.sg_loaded.emit()
            # Cascade reset
            self.subject = None
            self.object = None

    @property
    def subject(self) -> BoundingBox | None:
        return self._subject

    @subject.setter
    def subject(self, new_subject: BoundingBox | None):
        """Only emits if the new subject is different from the previous one."""
        old_subject = self._subject
        self._subject = new_subject
        if new_subject != old_subject:
            self.subject_changed.emit()

    @property
    def object(self) -> BoundingBox | None:
        return self._object

    @object.setter
    def object(self, new_object: BoundingBox | None):
        """Only emits if the new object is different from the previous one."""
        old_object = self._object
        self._object = new_object
        if new_object != old_object:
            self.object_changed.emit()

    def safe_remove_object_from_scene_graph(self, obj: BoundingBox):
        """
        Safely remove an object from a Scene Graph (assuming that the ParameterNodeHandler already did the work).
        Also remove any associated relation, unselect it and emit the right signals.
        """
        if self._scene_graph is None or obj is not self._scene_graph.get_bounding_box_by_id(obj.id):
            return

        for rel in list(self._scene_graph.iter_relations()):
            if rel.object_id == obj.id or rel.subject_id == obj.id:
                self._scene_graph.remove_relation(rel)
                self.relation_removed.emit(rel)

        self._scene_graph.remove_bounding_box(obj)
        # Update the subject/object if needed
        if self._subject == obj:
            self.subject = None
        if self._object == obj:
            self.object = None

        # Finally emit the removal
        self.sg_object_removed.emit(obj)


LOAD_STATE_MANAGER = LoadStateManager()
