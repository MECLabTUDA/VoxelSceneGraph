import qt

from scene_graph_api.knowledge import RelationRule
from scene_graph_api.scene import Relation
from ..load_state import LOAD_STATE_MANAGER
from .collapsible_button_widget import CollapsibleButtonWidget
from .relation_widget import RelationWidget


class RelationListingWidget(CollapsibleButtonWidget):
    """Collapsible widget with a list of relations widgets for a specific rule."""

    def __init__(self, relation_rule: RelationRule):
        super().__init__()

        self._rule_class = relation_rule

        # Connect to the relation list update
        LOAD_STATE_MANAGER.relation_added.connect(self._add_widget)
        # The widgets automatically commit to their destiny on deletion
        # LOAD_STATE_MANAGER.sg_object_removed.connect(self._update_object_combobox)

        self.init_ui()

    def init_ui(self):
        self.setText(self._rule_class.name)

        layout = qt.QVBoxLayout()
        self.setButtonLayout(layout)

        for rel in LOAD_STATE_MANAGER.scene_graph.relations_by_rule_id[self._rule_class.id]:
            self._add_widget(rel)

    def _add_widget(self, rel: Relation):
        """Add a widget for the relation."""
        if rel.rule_id == self._rule_class.id:
            self.button.layout().addWidget(RelationWidget(rel))
