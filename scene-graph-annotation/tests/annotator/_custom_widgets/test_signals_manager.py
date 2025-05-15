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

from unittest import TestCase
from unittest.mock import Mock

# noinspection PyProtectedMember
from scene_graph_annotation.annotator._custom_widgets import UIManager
from scene_graph_annotation.scene import Object


# noinspection DuplicatedCode
class TestSignalsManager(TestCase):
    def test_subject_changed(self):
        manager = UIManager()
        mock_callback = Mock()
        manager.subject_changed.connect(mock_callback)

        obj = Object(1, 1, "name", [], [])
        manager.subject = obj
        mock_callback.assert_called_once()
        # Should not trigger an event
        manager.subject = obj
        mock_callback.assert_called_once()
        # Should trigger an emit
        mock_callback = Mock()
        manager.subject_changed.connect(mock_callback)
        manager.subject = Object(2, 1, "name", [], [])
        mock_callback.assert_called_once()

    def test_object_changed(self):
        manager = UIManager()
        mock_callback = Mock()
        manager.object_changed.connect(mock_callback)

        obj = Object(1, 1, "name", [], [])
        manager.object = obj
        mock_callback.assert_called_once()
        # Should not trigger an event
        manager.object = obj
        mock_callback.assert_called_once()
        # Should trigger an emit
        mock_callback = Mock()
        manager.object_changed.connect(mock_callback)
        manager.object = Object(2, 1, "name", [], [])
        mock_callback.assert_called_once()
