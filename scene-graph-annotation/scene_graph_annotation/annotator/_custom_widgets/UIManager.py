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

from PyQt6.QtCore import pyqtSignal, QObject

from scene_graph_annotation.scene import BoundingBox, Attribute, Relation


class UIManager(QObject):
    """
    Class used to centralize all signals, options, ... related to the UI: one manager FOR EACH scene graph.
    Used to store references to the subject/object (if selected) and to signals for any update.
    Widgets should keep a reference to this object for retrieving or setting subjects/objects.
    Also used to have a centralized object for object name/attribute change.
    Also used for object creation/deletion e.g. during merge/split.
    """
    # Signals that the user selected another subject
    subject_changed = pyqtSignal()
    # Signals that the user selected another object
    object_changed = pyqtSignal()

    # Signals that the attributes name has changed
    # We cannot have a signal per object as signals need to be class attributes
    # So we pass the object as argument and let callees figure out if they need to do something
    name_changed = pyqtSignal(BoundingBox)
    # Same but with attributes
    attribute_changed = pyqtSignal(BoundingBox, Attribute)
    # Same but with a Bounding Box or Segmentation creation
    sg_object_added = pyqtSignal(BoundingBox)
    # Same but with a Bounding Box or Segmentation deletion
    sg_object_removed = pyqtSignal(BoundingBox)
    # Same but with a relation creation
    relation_added = pyqtSignal(Relation)
    # Same but with a relation deletion
    relation_removed = pyqtSignal(Relation)

    # Signals used to select the "correct" slice to show the object with the given id to the user
    show_object = pyqtSignal(int)

    def __init__(self):
        super().__init__()

        self._subject: BoundingBox | None = None
        self._object: BoundingBox | None = None

    @property
    def subject(self) -> BoundingBox | None:
        return self._subject

    @property
    def object(self) -> BoundingBox | None:
        return self._object

    @subject.setter
    def subject(self, new_subject: BoundingBox | None):
        """Only emits if the new subject is different from the previous one."""
        old_subject = self._subject
        self._subject = new_subject
        if new_subject != old_subject:
            self.subject_changed.emit()

    @object.setter
    def object(self, new_object: BoundingBox | None):
        """Only emits if the new object is different from the previous one."""
        old_object = self._object
        self._object = new_object
        if new_object != old_object:
            self.object_changed.emit()
