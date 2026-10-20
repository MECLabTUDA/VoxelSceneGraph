import qt
from PythonQt.QtGui import QSizePolicy

from scene_graph_api.scene import BoundingBox
from .collapsible_button_widget import CollapsibleButtonWidget
from .object_editor import ObjectEditorWidget
from .object_selector import ObjectSelector
from .toggle import ResponsiveToggle
from ..load_state import LOAD_STATE_MANAGER


class DynamicStackedWidget(qt.QStackedWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Connect the page change signal to trigger a layout recalculation
        self.currentChanged.connect(self.update_geometry_on_change)

    def update_geometry_on_change(self, index):
        # Tell the layout system that the size hints have changed
        self.updateGeometry()

    def sizeHint(self):
        current_widget = self.currentWidget()
        if current_widget:
            # Return the size hint of ONLY the active widget
            return current_widget.sizeHint

    def minimumSizeHint(self):
        current_widget = self.currentWidget()
        if current_widget:
            # Prevent the container from shrinking smaller than the active widget's minimum
            return current_widget.minimumSizeHint


class ObjectsTab(qt.QWidget):
    """
    Tab widget that contains attribute annotation for both image-level and objects.
    With the compact layout, we only have a single object editor at a time, the object being selected in a combobox.
    Otherwise, have a bunch of editors.
    """
    _compact_layout = True  # Keep the state here rather than in the LOAD_STATE_MANAGER for simplicity

    def __init__(self):
        super().__init__()

        self._current_compact_editor: qt.QWidget | None = None
        self._expanded_editors_layout: qt.QVBoxLayout | None = None

        # Useful we ever need to delete widgets (e.g. with segmentations)
        self._object_editor_by_id: dict[int, qt.QWidget] = {}

        # Connect to the study list update
        LOAD_STATE_MANAGER.sg_loaded.connect(self._update_ui)
        LOAD_STATE_MANAGER.sg_object_added.connect(self.add_object)
        LOAD_STATE_MANAGER.sg_object_removed.connect(self.remove_object)

    def _update_ui(self):
        """Sets the state of form widgets."""
        layout = self.layout()
        if layout is not None:
            self._object_editor_by_id = {}
            if self._current_compact_editor is not None:
                try:
                    self._current_compact_editor.deleteLater()
                except ValueError:
                    # Widget destroyed already
                    ...
                self._current_compact_editor = None
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
        else:
            layout = qt.QFormLayout()
            self.setLayout(layout)

        scene_graph = LOAD_STATE_MANAGER.scene_graph
        if scene_graph is None:
            # Only reset the layout
            return

        # Toggle between compact and expanded layout
        compact_layout_toggle = ResponsiveToggle()
        layout.addRow(qt.QLabel("Compact Layout:"), compact_layout_toggle)
        compact_layout_toggle.setChecked(self._compact_layout)

        # Image level
        knowledge_graph = scene_graph.knowledge_graph
        image_level_widget = ObjectEditorWidget(
            knowledge_graph.image,
            scene_graph.image,
            is_image_level=True
        )
        layout.addRow(image_level_widget)

        # Object editors
        stacked_editors = DynamicStackedWidget()
        layout.addRow(stacked_editors)

        def _toggle(checked):
            checked = bool(checked)  # Convert from CheckState
            self._compact_layout = checked
            stacked_editors.setCurrentIndex(int(checked))

        compact_layout_toggle.stateChanged.connect(_toggle)

        # Expanded layout
        expanded_editors_widget = qt.QWidget()
        stacked_editors.addWidget(expanded_editors_widget)
        self._expanded_editors_layout = qt.QVBoxLayout()
        expanded_editors_widget.setLayout(self._expanded_editors_layout)

        # All objects no matter the class
        for obj in scene_graph.iter_bounding_boxes(sort=True):
            editor = ObjectEditorWidget(
                knowledge_graph.get_object_class_by_id(obj.class_id),
                obj,
                common_object_class=knowledge_graph.obj_common,
                is_image_level=False
            )
            self._expanded_editors_layout.addWidget(editor)
            self._object_editor_by_id[obj.id] = editor

        # Compact layout
        compact_editors_widget = CollapsibleButtonWidget()
        compact_editors_widget.setText("Objects")
        stacked_editors.addWidget(compact_editors_widget)
        compact_layout = qt.QFormLayout()
        compact_editors_widget.setButtonLayout(compact_layout)
        # Combobox for object selection
        object_selector = ObjectSelector()
        compact_layout.addRow(qt.QLabel("Object:"), object_selector)

        # Init state
        _toggle(self._compact_layout)

        def _swap_compact_editor():
            """Swaps the current compact editor with the current expanded editor."""
            if self._current_compact_editor is not None:
                try:
                    self._current_compact_editor.deleteLater()
                except ValueError:
                    # Already deleted
                    ...
            obj_ = object_selector.currentData
            if obj_ is not None:
                self._current_compact_editor = ObjectEditorWidget(
                    knowledge_graph.get_object_class_by_id(obj_.class_id),
                    obj_,
                    common_object_class=knowledge_graph.obj_common,
                    is_image_level=False
                )
                compact_layout.addRow(self._current_compact_editor)

        object_selector.currentIndexChanged.connect(_swap_compact_editor)

    def add_object(self, obj: BoundingBox):
        """Adds a widget for the object in the expanded layout."""
        scene_graph = LOAD_STATE_MANAGER.scene_graph
        if scene_graph is None:
            return

        knowledge_graph = scene_graph.knowledge_graph
        editor = ObjectEditorWidget(
            knowledge_graph.get_object_class_by_id(obj.class_id),
            obj,
            common_object_class=knowledge_graph.obj_common,
            is_image_level=False
        )

        self._object_editor_by_id[obj.id] = editor
        self._expanded_editors_layout.addWidget(editor)

    def remove_object(self, obj: BoundingBox):
        """Deletes the widget associated with the associated object."""
        editor = self._object_editor_by_id.get(obj.id)
        if editor is None:
            return
        del self._object_editor_by_id[obj.id]
        editor.deleteLater()
