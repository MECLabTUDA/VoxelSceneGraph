import qt
from PythonQt.QtGui import QColor, QPixmap, QIcon

from scene_graph_api.scene import BoundingBox
from ..load_state import LOAD_STATE_MANAGER


class ObjectSelector(qt.QComboBox):
    """
    QComboBox with a list of all objects in the Scene Graph.
    Item 0 is no object selected (None).
    """

    def __init__(self):
        super().__init__()
        # FIXME Connect not triggering for some fucking reason
        # LOAD_STATE_MANAGER.sg_loaded.connect(self.update_object_list)
        LOAD_STATE_MANAGER.sg_object_added.connect(self.update_object_list_with_keep_selection)
        LOAD_STATE_MANAGER.sg_object_removed.connect(self.update_object_list_with_keep_selection)
        LOAD_STATE_MANAGER.update_color.connect(self._update_colors)

        self.update_object_list()

    @staticmethod
    def _get_icon(color: str | None):
        """Returns an icon for the given object."""
        size = 16
        pixmap = QPixmap(size, size)
        pixmap.fill(QColor(color))
        return QIcon(pixmap)

    def _update_colors(self):
        """Updates the icon colors."""
        # I'm lazy, so just update all icons
        try:
            for idx, obj in enumerate(LOAD_STATE_MANAGER.scene_graph.iter_bounding_boxes(sort=True)):
                icon = self._get_icon(LOAD_STATE_MANAGER.colors.get(obj.id))
                self.setItemIcon(idx + 1, icon)
        except ValueError:
            # Object already destroyed
            return


    def update_object_list(self):
        """Refresh the list of objects."""

        self.clear()
        if LOAD_STATE_MANAGER.scene_graph is None:
            return

        self.addItem("", None)
        for idx, obj in enumerate(LOAD_STATE_MANAGER.scene_graph.iter_bounding_boxes(sort=True)):
            self.addItem(obj.name, obj)

            icon = self._get_icon(LOAD_STATE_MANAGER.colors.get(obj.id))
            self.setItemIcon(idx + 1, icon)


    def update_object_list_with_keep_selection(self):
        """
        Refresh the list of objects.
        Select the same object if still in the Scene Graph.
        Otherwise, select the next object.
        """
        if LOAD_STATE_MANAGER.scene_graph is None:
            return

        try:
            prev = self.currentData
        except ValueError:
            # Object already destroyed
            return

        self.update_object_list()

        # Also stay on the first item, if nothing is selected
        if prev is None or LOAD_STATE_MANAGER.scene_graph.get_bounding_box_by_id(prev.id) is prev:
            self.select_object(prev)
        else:
            # Iterate over all objects to find the next one in the list
            for obj in LOAD_STATE_MANAGER.scene_graph.iter_bounding_boxes(sort=True):
                if obj.id >= prev.id:
                    self.select_object(obj)
                    break
            else:
                self.select_object(None)

    def select_object(self, new_obj: BoundingBox | None):
        """Selects the given object."""
        if new_obj is None:
            self.setCurrentIndex(0)
            return

        for idx, obj in enumerate(LOAD_STATE_MANAGER.scene_graph.iter_bounding_boxes(sort=True)):
            if obj == new_obj:
                self.setCurrentIndex(idx + 1)
                break

    def _disconnect_signals(self):
        """Disconnects all signals."""
        # LOAD_STATE_MANAGER.sg_loaded.disconnect(self.update_object_list)
        LOAD_STATE_MANAGER.sg_object_added.disconnect(self.update_object_list)
        LOAD_STATE_MANAGER.sg_object_removed.disconnect(self.update_object_list)
        LOAD_STATE_MANAGER.update_color.disconnect(self._update_colors)
