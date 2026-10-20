from __future__ import annotations


try:
    import sqlite_utils
    import referencing
    import jsonschema
    import jsonpointer
    import cc3d
    import nibabel
    import tqdm
except ImportError:
    import slicer
    import qt

    input_dialog = qt.QInputDialog()
    input_dialog.setOptions(qt.QInputDialog.UsePlainTextEditForTextInput)
    proxy = input_dialog.getItem(
        slicer.util.mainWindow(),  # Parent widget
        "Python Library Installation",  # Window Title
        "We wish to install additional Python libraries. "
        "Please enter the proxy information for pip. "
        "Leave it empty, if there is no proxy.",
        [],
        0,
        True
    )

    modules = [
        "sqlite-utils",
        "referencing",
        "jsonschema",
        "jsonpointer",
        "connected-components-3d",
        "nibabel",
        "tqdm"
    ]
    for module in modules:
        if proxy:
            slicer.util.pip_install([module, "--proxy", proxy])
        else:
            slicer.util.pip_install(module)
    import sqlite_utils

import qt

import vtkSegmentationCore
import slicer
import vtk
from PythonQt.QtCore import Qt
from slicer import vtkMRMLScalarVolumeNode, vtkMRMLSegmentationNode

from slicer.util import VTKObservationMixin

from slicer.ScriptedLoadableModule import *

from slicer.i18n import tr as _
from slicer.i18n import translate




class SceneGraphAnnotationOverview(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Scene Graph Annotation")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Segmentation")]
        self.parent.dependencies = []
        self.parent.contributors = ["Antoine P. Sanner (University Hospital Mayence, Germany)"]
        self.parent.helpText = _("""Easily load and annotate patients for a study. 
        Shortcuts:<br>
        -s: save the progress for the selected patient (if any)<br>
        -e: select the previous patient (if a patient is already selected)<br>
        -r: select the previous patient (if a patient is already selected)<br>
        -c: if there is a segment under the cursor, select it as the current segment in the segment editor
        """)
        self.parent.acknowledgementText = _("")


class SceneGraphAnnotationOverviewWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Widget for easy patient list annotation and overview."""

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        from scene_graph_utils.widgets.config_widget import ConfigWidget
        from scene_graph_utils.widgets.tools_widget import ToolsWidget
        from scene_graph_utils.nodes import ParameterNodeHandler

        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation

        self._segment_editor = slicer.modules.segmenteditor.widgetRepresentation()
        self._parameter_node_handler = ParameterNodeHandler(self._segment_editor, node=None)
        self._parameter_node_gui_tag = None

        self._cfg_widget = ConfigWidget()
        self._tools_widget = ToolsWidget(self._parameter_node_handler, self._segment_editor)

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()
        self.setupUI()
        self._install_keyboard_shortcuts()

    def setupUI(self):
        # Make some room
        # slicer.util.setModuleHelpSectionVisible(False)

        # ==============================================================================================================
        self.layout.addWidget(self._cfg_widget)
        self.layout.addWidget(self._tools_widget)

        # ==============================================================================================================
        # Add some small stretch to allow the data probe widget to expand without creating a scrollbar
        # when hovering over a segment with the cursor
        # Note: the tools QTabWidget has a minimum height set from one of the tools widget that we add
        self.layout.addStretch(1)

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # The user might have visited the data, segment editor, volumes module, which might have reset the layout
        self._tools_widget.setup_tabs()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        from scene_graph_utils.load_state import LOAD_STATE_MANAGER
        from scene_graph_utils.nodes import SceneGraphAnnotationOverviewParameterNode

        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.
        param_node = slicer.mrmlScene.GetSingletonNode(self.moduleName, "vtkMRMLScriptedModuleNode")
        if param_node is None:
            param_node = slicer.mrmlScene.CreateNodeByClass("vtkMRMLScriptedModuleNode")
            param_node.UnRegister(None)  # object is owned by the Python variable now
            param_node.SetSingletonTag(self.moduleName)
            # Add module name in an attribute to allow filtering in node selector widgets
            # Note that SetModuleName is not used anymore as it would be redundant with the ModuleName attribute.
            param_node.SetAttribute("ModuleName", self.moduleName)
            param_node.SetName(slicer.mrmlScene.GenerateUniqueName(self.moduleName))
            slicer.mrmlScene.AddNode(param_node)

        self.setParameterNode(SceneGraphAnnotationOverviewParameterNode(param_node))
        LOAD_STATE_MANAGER.selected_patient_name = None

    def setParameterNode(self, inputParameterNode) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        parameter_node = self._parameter_node_handler.node
        if parameter_node:
            parameter_node.disconnectGui(self._parameter_node_gui_tag)

        self._parameter_node_handler.node = inputParameterNode

    def _install_keyboard_shortcuts(self):
        """Install some additional keybind shortcuts. These are always active when the module is open."""
        from scene_graph_utils.load_state import LOAD_STATE_MANAGER
        self._shortcuts = []

        # Select previous patient
        prev_patient_shortcut = qt.QShortcut(self.parent)
        prev_patient_shortcut.setKey(qt.QKeySequence("e"))
        prev_patient_shortcut.connect("activated()", lambda: LOAD_STATE_MANAGER.select_patient_by_offset.emit(-1))
        self._shortcuts.append(prev_patient_shortcut)

        # Select next patient
        next_patient_shortcut = qt.QShortcut(self.parent)
        next_patient_shortcut.setKey(qt.QKeySequence("r"))
        next_patient_shortcut.connect("activated()", lambda: LOAD_STATE_MANAGER.select_patient_by_offset.emit(1))
        self._shortcuts.append(next_patient_shortcut)

        # Switch the value windowing for the volume
        switch_windowing_shortcut = qt.QShortcut(self.parent)
        switch_windowing_shortcut.setKey(qt.QKeySequence("v"))
        switch_windowing_shortcut.connect("activated()", self._parameter_node_handler.show_next_image_windowing)
        self._shortcuts.append(switch_windowing_shortcut)

        # Select segment under the cursor
        select_segment_cursor_shortcut = qt.QShortcut(self.parent)
        select_segment_cursor_shortcut.setKey(qt.QKeySequence("c"))
        select_segment_cursor_shortcut.connect("activated()", self._parameter_node_handler.select_segment_under_cursor)
        self._shortcuts.append(select_segment_cursor_shortcut)
