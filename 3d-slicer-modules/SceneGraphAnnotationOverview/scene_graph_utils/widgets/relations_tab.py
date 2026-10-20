import qt

from ..load_state import LOAD_STATE_MANAGER
from .add_relation_widget import AddRelationWidget
from .relation_listing_widget import RelationListingWidget


class RelationsTab(qt.QWidget):
    """Tab widget that contains relation annotation."""

    def __init__(self):
        super().__init__()

        # Connect to the study list update
        LOAD_STATE_MANAGER.sg_loaded.connect(self._update_ui)
        self.destroyed.connect(self._disconnect_signals)

    def _update_ui(self):
        """Sets the state of widgets."""
        layout = self.layout()
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
        else:
            layout = qt.QVBoxLayout()
            self.setLayout(layout)

        scene_graph = LOAD_STATE_MANAGER.scene_graph
        if scene_graph is None:
            # Only reset the layout
            return

        # Add relation widget
        layout.addWidget(AddRelationWidget())

        # Relation listing widgets
        for relation_rule in LOAD_STATE_MANAGER.scene_graph.knowledge_graph.rules:
            layout.addWidget(RelationListingWidget(relation_rule))
        layout.addStretch(1)

    def _disconnect_signals(self):
        """Disconnects all signals from the widget."""
        LOAD_STATE_MANAGER.sg_loaded.disconnect(self._update_ui)
