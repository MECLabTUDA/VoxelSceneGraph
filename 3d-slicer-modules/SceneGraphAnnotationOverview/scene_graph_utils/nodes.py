"""Most objects related to 3D Slicer fluff."""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import qt
import slicer
import vtk
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, inv_ornt_aff
from slicer import vtkMRMLScalarVolumeNode, vtkMRMLSegmentationNode
from slicer.parameterNodeWrapper import parameterNodeWrapper

from scene_graph_api.knowledge import RadiologyImageKG
from scene_graph_api.scene import SceneGraph, BoundingBox
from scene_graph_api.utils.array_utils import bbox_2d, bbox_3d
from scene_graph_api.utils.pathing import remove_suffixes
from scene_graph_database.options import Options
from scene_graph_database.structures import StudyStructure
from scene_graph_utils.load_state import LOAD_STATE_MANAGER


@parameterNodeWrapper
class SceneGraphAnnotationOverviewParameterNode:
    """
    The parameters needed by the module.
    Represents a selected patient (i.e. a volume and a segmentation).
    Patient name and study can be retrieved from widgets.
    """
    # Has to use "Optional" instead of "| None" because of slicer typing issues for serialization
    volume: Optional[vtkMRMLScalarVolumeNode]
    preview_volume: Optional[vtkMRMLScalarVolumeNode]
    segmentation: Optional[vtkMRMLSegmentationNode]
    ai_segmentation: Optional[vtkMRMLSegmentationNode]

    # noinspection PyUnresolvedReferences
    def reset(self):
        """Reset the parameter node to its initial state. And remove nodes from the scene."""
        # Remove nodes from the scene
        if self.volume is not None:
            slicer.mrmlScene.RemoveNode(self.volume)
        if self.preview_volume is not None:
            slicer.mrmlScene.RemoveNode(self.preview_volume)
        if self.segmentation is not None:
            slicer.mrmlScene.RemoveNode(self.segmentation)
        if self.ai_segmentation is not None:
            slicer.mrmlScene.RemoveNode(self.ai_segmentation)
        # Set new values
        self.setValue("volume", None)
        self.setValue("preview_volume", None)
        self.setValue("segmentation", None)
        self.setValue("ai_segmentation", None)


class ParameterNodeHandler:
    """Used to automatically load relevant volumes and do the relevant processing."""

    def __init__(
            self,
            segment_editor,
            node: SceneGraphAnnotationOverviewParameterNode | None = None,
    ):
        self._segment_editor = segment_editor
        self._node = node

        # State for image windowing
        self._current_windowing_index = 0
        # State for segment and SG object mapping
        self._segment_id_to_sg_object = {}
        self._sg_object_to_segment_id = {}

        # Connect to signals
        LOAD_STATE_MANAGER.patient_selected.connect(self.setup_patient)
        LOAD_STATE_MANAGER.preview_toggled.connect(self._setup_volume)
        LOAD_STATE_MANAGER.ai_prediction_toggled.connect(self._toggle_ai_prediction)
        LOAD_STATE_MANAGER.name_changed.connect(self._handle_name_change)
        LOAD_STATE_MANAGER.show_object.connect(self._handle_center_views_on_object)
        LOAD_STATE_MANAGER.remove_segment_for_object.connect(self.external_segment_delete)

        # Fix for delete post segment add
        self._timer = qt.QTimer()
        self._timer.setInterval(.1)
        self._timer.setSingleShot(True)

    def setup_patient(self):
        """Load the Scene Graph. Warning: if some loading failed, the selected patient name is reverted to None."""
        if LOAD_STATE_MANAGER.selected_patient_name is None:
            # We cleared the selection
            self.reset()
            return

        if LOAD_STATE_MANAGER.knowledge_graph is None:
            print("setup_patient: No knowledge graph loaded")
            # We need to cancel the selection
            LOAD_STATE_MANAGER.selected_patient_name = None
            LOAD_STATE_MANAGER.scene_graph = None
            return
        if LOAD_STATE_MANAGER.selected_study_idx is None:
            print("setup_patient: No study loaded")
            # We need to cancel the selection
            LOAD_STATE_MANAGER.selected_patient_name = None
            LOAD_STATE_MANAGER.scene_graph = None
            return

        # Try to load the Scene Graph
        structure = StudyStructure(
            Path(Options.get_db_string()),
            LOAD_STATE_MANAGER.selected_study.knowledge_graph_path
        )

        logger = logging.getLogger(__name__)
        scene_graph = SceneGraph.load(
            structure.scene_graphs / (remove_suffixes(Path(LOAD_STATE_MANAGER.selected_patient_name)) + ".json"),
            LOAD_STATE_MANAGER.knowledge_graph,
            logger
        )

        # On error, abort to unselection
        if scene_graph is None:
            slicer.util.confirmOkCancelDisplay(
                f"Could not open the Scene Graph",
                windowTitle="Error", parent=None
            )
            LOAD_STATE_MANAGER.selected_patient_name = None
            # FIXME order of triggers for scene graph loading vs patient name selection
            LOAD_STATE_MANAGER.scene_graph = None
            return

        # We succeeded, so we can update the states
        self.reset()

        self._setup_volume()
        self.update_image_windowing()
        self._setup_segmentation(scene_graph)

        LOAD_STATE_MANAGER.selected_patient_scene_graph_modified = False
        LOAD_STATE_MANAGER.scene_graph = scene_graph

        # For each segment also trigger the color change
        self._handle_segment_modified(None, None)

        # Attempt to load the AI segmentation / labelmap
        ai_pred_path = structure.ai_predictions / LOAD_STATE_MANAGER.selected_patient_name
        if ai_pred_path.is_file():
            try:
                ai_seg_node = slicer.util.loadSegmentation(ai_pred_path.as_posix())
                # TODO if there is a corresponding JSON file (i.e. labelmap), then we should rename segments
                self._node.ai_segmentation = ai_seg_node
                # Handles toggling the visibility of the AI segmentation
                self._toggle_ai_prediction()
            except RuntimeError:
                slicer.util.confirmOkCancelDisplay(
                    f"Could not read the AI segmentation: {ai_pred_path}",
                    windowTitle="Error", parent=None
                )

    @property
    def volume(self) -> vtkMRMLScalarVolumeNode | None:
        return self._node.volume

    @property
    def preview_volume(self) -> vtkMRMLScalarVolumeNode | None:
        return self._node.preview_volume

    @property
    def segmentation(self) -> vtkMRMLScalarVolumeNode | None:
        return self._node.segmentation

    @property
    def ai_segmentation(self) -> vtkMRMLScalarVolumeNode | None:
        return self._node.ai_segmentation

    @property
    def node(self) -> SceneGraphAnnotationOverviewParameterNode:
        return self._node

    @node.setter
    def node(self, node):
        """Set the parameter node. Requires a reset."""
        self.reset()
        self._node = node

    def _setup_volume(self) -> bool:
        """Toggle the preview volume."""
        if self._node is None:
            print("toggle_preview: No parameter node loaded")
            return False
        if LOAD_STATE_MANAGER.selected_study_idx is None:
            print("toggle_preview: No study loaded")
            return False
        if LOAD_STATE_MANAGER.selected_patient_name is None:
            print("toggle_preview: No patient selected")
            return False

        # Check that everything is here and can be loaded
        # Note: we check that the full volume is at least there no matter what
        structure = StudyStructure(
            Path(Options.get_db_string()),
            LOAD_STATE_MANAGER.selected_study.knowledge_graph_path
        )

        preview_failed = False
        # Handle preview case
        if LOAD_STATE_MANAGER.show_preview:
            if self._node.preview_volume is not None:
                # Select/show it in viewers
                selection_node = slicer.app.applicationLogic().GetSelectionNode()
                selection_node.SetActiveVolumeID(self._node.preview_volume.GetID())
                slicer.app.applicationLogic().PropagateVolumeSelection(0)
                return True

            # We need to load the volume
            try:
                vol_node = slicer.util.loadVolume(
                    (structure.images_preview / LOAD_STATE_MANAGER.selected_patient_name).as_posix()
                )
                self._node.preview_volume = vol_node
                self.update_image_windowing()
                return True
            except RuntimeError as e:
                # If for any reason, we cannot load the preview volume, just switch to the full volume
                preview_failed = True

        # Handle the normal case
        if not LOAD_STATE_MANAGER.show_preview or preview_failed:
            if self._node.volume is not None:
                # Select/show it in viewers
                selection_node = slicer.app.applicationLogic().GetSelectionNode()
                selection_node.SetActiveVolumeID(self._node.volume.GetID())
                slicer.app.applicationLogic().PropagateVolumeSelection(0)
                return True

            # We need to load the volume
            vol_path = structure.images / LOAD_STATE_MANAGER.selected_patient_name
            try:
                vol_node = slicer.util.loadVolume(vol_path.as_posix())
                self._node.volume = vol_node
                self.update_image_windowing()
                return True
            except RuntimeError:
                # If for any reason, we cannot load the preview volume, just switch to the full volume
                slicer.util.confirmOkCancelDisplay(
                    f"Could not read the volume: {vol_path}",
                    windowTitle="Error", parent=None
                )
                LOAD_STATE_MANAGER.selected_patient_name = None
                return False
            finally:
                # Just in case the segmentation was set up already, set the source volume
                if self._node.segmentation is not None and self._node.volume is not None:
                    self._segment_editor.self().editor.setSourceVolumeNode(self._node.volume)
                # Set the volume in the "Volumes" module
                slicer.modules.volumes.widgetRepresentation().setEditedNode(self._node.volume)

        # We should never end up here
        return False

    def _setup_segmentation(self, scene_graph: SceneGraph):
        """
        Loads the segmentation and adds hooks, renames segments.
        The master volume is only set if not None.
        Note: we need to pass the Scene Graph as an argument because we do not have set it yet in the LOAD_STATE_MANAGER.
        """
        if self._node is None:
            print("setup_segmentation: No parameter node loaded")
            return

        # Create segmentation node
        seg_node = slicer.vtkMRMLSegmentationNode()
        slicer.mrmlScene.AddNode(seg_node)
        seg_node.CreateDefaultDisplayNodes()  # only needed for display

        modified_event = seg_node.StartModify()

        # Use a labelmap volume for bulk import
        labelmap_vol = slicer.vtkMRMLLabelMapVolumeNode()
        # Set geometry
        ras_affine = self._get_reoriented_affine(scene_graph)
        ijk_ras = slicer.util.vtkMatrixFromArray(ras_affine)
        labelmap_vol.SetIJKToRASMatrix(ijk_ras)

        # Trick: replace background with a placeholder label filling the entire image
        tmp = scene_graph.object_labelmap.copy()
        use_trick = np.any(tmp == 0)
        # Only use the trick if there is any background (if so ok, would cause issues if the dummy segment is empty)
        if use_trick:
            tmp_id = scene_graph.get_next_available_bounding_box_id()
            tmp[tmp == 0] = tmp_id
        slicer.util.updateVolumeFromArray(labelmap_vol, tmp)
        seg_node.SetReferenceImageGeometryParameterFromVolumeNode(labelmap_vol)

        # Convert from labelmap to segmentation
        seg_logic = slicer.modules.segmentations.logic()
        # Note: this dumb function automatically crops the internal array representation.
        #       So to keep it the size of the image, we have to add a placeholder label
        seg_logic.ImportLabelmapToSegmentationNode(labelmap_vol, seg_node)

        # Create a new segment for each scene graph object
        segmentation = seg_node.GetSegmentation()
        n_segments = segmentation.GetNumberOfSegments()
        assert n_segments == len(list(scene_graph.iter_bounding_boxes(sort=True))) + int(use_trick)
        for i, obj in zip(range(n_segments), scene_graph.iter_bounding_boxes(sort=True)):
            segment_id = segmentation.GetNthSegmentID(i)
            segment = segmentation.GetSegment(segment_id)

            # Update Name
            segment.SetName(obj.name)
            # Update mappings
            self._segment_id_to_sg_object[segment_id] = obj
            self._sg_object_to_segment_id[obj.id] = segment_id

        # If the trick was used, remove the placeholder label
        if use_trick:
            segmentation.RemoveSegment(segmentation.GetNthSegmentID(n_segments - 1))

        del labelmap_vol

        seg_node.EndModify(modified_event)
        self._node.segmentation = seg_node

        # This updates all other modules (e.g. Data). Somehow it also makes the first segment go invisible...
        seg_node.Modified()

        # Set the volumes in the segment editor
        self._segment_editor.self().editor.setSegmentationNode(self._node.segmentation)
        if not LOAD_STATE_MANAGER.show_preview:
            self._segment_editor.self().editor.setSourceVolumeNode(self._node.volume)
        if segmentation.GetNumberOfSegments() > 0:
            self._segment_editor.self().editor.setCurrentSegmentID(segmentation.GetNthSegmentID(0))

        # Add hooks
        self._setup_segmentation_hooks()

    def _setup_segmentation_hooks(self):
        """Add observers for the segmentation."""
        if self._node is None:
            print("setup_segmentation_hooks: No parameter node loaded")
            return
        seg_node = self._node.segmentation
        if seg_node is None:
            print("setup_segmentation_hooks: No segmentation loaded")
            return

        if seg_node is not None:
            seg_node.AddObserver(
                seg_node.GetSegmentation().SegmentAdded,
                self._handle_segment_added
            )
            seg_node.AddObserver(
                seg_node.GetSegmentation().SegmentModified,
                self._handle_segment_modified
            )
            seg_node.AddObserver(
                seg_node.GetSegmentation().SegmentRemoved,
                self._handle_segment_removed
            )

    @staticmethod
    def _get_reoriented_affine(scene_graph: SceneGraph) -> np.ndarray:
        """3D Slicer expects the affine matrix to be in the opposite order for some reason, so dix this here."""
        ornt_current = io_orientation(scene_graph.image_affine)
        ornt_target = ornt_current[::-1]  # Reordering required by 3D Slicer
        transform = ornt_transform(ornt_current, ornt_target)
        return scene_graph.image_affine.dot(inv_ornt_aff(transform, scene_graph.object_labelmap.shape))

    @staticmethod
    def _get_segmentation_ref_volume_from_scene_graph(scene_graph: SceneGraph):
        """Get a reference volume for the segmentation with the same geometry."""
        tmp = slicer.vtkMRMLScalarVolumeNode()
        slicer.util.updateVolumeFromArray(tmp, scene_graph.object_labelmap)
        ijk_ras = slicer.util.vtkMatrixFromArray(ParameterNodeHandler._get_reoriented_affine(scene_graph))
        tmp.SetIJKToRASMatrix(ijk_ras)
        return tmp

    def save_all_annotations(self):
        """Save all annotations in the Scene Graph."""
        scene_graph = LOAD_STATE_MANAGER.scene_graph
        if scene_graph is None:
            print("save_all_annotations: No scene graph loaded")
            return
        seg_node = self._node.segmentation
        if seg_node is None:
            print("save_all_annotations: No segmentation loaded")
            return

        # Update the label map and recompute bounding boxes
        # Note: We should be keeping 1-to-1 mappings between scene graph objects and segmentation segments
        #       This means that any object still in the Scene Graph has a segmentation counterpart
        # Note: Attribute, key point and relation annotations are already saved in place in the Scene Graph
        segmentation = seg_node.GetSegmentation()
        new_labelmap = np.zeros_like(scene_graph.object_labelmap)
        n_dim = len(new_labelmap.shape)
        # Note: Iterate this way so that we are sure to respect the segment ordering shown in the editor
        #       Otherwise it would result in unexpected behavior in case of segment overlap
        tmp = self._get_segmentation_ref_volume_from_scene_graph(scene_graph)
        for idx in range(segmentation.GetNumberOfSegments()):
            segment_uid = segmentation.GetNthSegmentID(idx)
            obj = self._segment_id_to_sg_object[segment_uid]
            segment_arr = slicer.util.arrayFromSegmentBinaryLabelmap(seg_node, segment_uid, tmp)
            # Check that the mask is not empty
            if not np.any(segment_arr):
                scene_graph.remove_bounding_box(obj)
                continue
            new_labelmap[segment_arr > 0] = obj.id
            obj.bounding_box = bbox_3d(segment_arr) if n_dim == 3 else bbox_2d(segment_arr)
        scene_graph.object_labelmap = new_labelmap
        del tmp

        # Save the final Scene Graph
        structure = StudyStructure(
            Path(Options.get_db_string()),
            LOAD_STATE_MANAGER.selected_study.knowledge_graph_path
        )
        sg_path = structure.scene_graphs / (remove_suffixes(Path(LOAD_STATE_MANAGER.selected_patient_name)) + ".json")
        scene_graph.save(sg_path)
        LOAD_STATE_MANAGER.selected_patient_scene_graph_modified = False

    def show_next_image_windowing(self):
        """Show the next image windowing range."""
        knowledge_graph = LOAD_STATE_MANAGER.knowledge_graph
        if isinstance(knowledge_graph, RadiologyImageKG) and \
                len(knowledge_graph.window_center) > 0 and len(knowledge_graph.window_width) > 0:
            self._current_windowing_index = (self._current_windowing_index + 1) % len(knowledge_graph.window_center)
            self.update_image_windowing()

    def update_image_windowing(self):
        """Update the image windowing of loaded images."""
        knowledge_graph = LOAD_STATE_MANAGER.knowledge_graph
        if self._node is not None and isinstance(knowledge_graph, RadiologyImageKG) and \
                len(knowledge_graph.window_center) > 0 and len(knowledge_graph.window_width) > 0:
            center = knowledge_graph.window_center[self._current_windowing_index]
            width = knowledge_graph.window_width[self._current_windowing_index]
            mmin = center - width / 2
            mmax = center + width / 2

            vol_node = self._node.volume
            if vol_node is not None:
                vol_node.GetDisplayNode().SetAutoWindowLevel(False)
                vol_node.GetDisplayNode().SetWindowLevelMinMax(mmin, mmax)
            preview_vol_node = self._node.preview_volume
            if preview_vol_node is not None:
                preview_vol_node.GetDisplayNode().SetAutoWindowLevel(False)
                preview_vol_node.GetDisplayNode().SetWindowLevelMinMax(mmin, mmax)

    def reset(self):
        """Reset the state of the parameter node."""
        if self._node is not None:
            self._node.reset()
        self._current_windowing_index = 0
        self._segment_id_to_sg_object = {}
        self._sg_object_to_segment_id = {}

    def select_segment_under_cursor(self):
        """
        Try to find out if there is a segment under the cursor.
        If it belongs to the current segmentation, select this segment as the current segment in the segment editor.
        """
        if self._node is None or self._node.segmentation is None:
            # No current segmentation
            return

        # Find out the position of the cursor
        crosshair_node = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLCrosshairNode")
        if crosshair_node is None:
            return

        ras_pos = [0, 0, 0]
        inside_view = crosshair_node.GetCursorPositionRAS(ras_pos)
        xyz_pos = [0, 0, 0]
        slice_node = crosshair_node.GetCursorPositionXYZ(xyz_pos)
        if not inside_view or not slice_node:
            return

        slice_widget = slicer.app.layoutManager().sliceWidget(slice_node.GetName())
        seg_disp_manager = slice_widget.sliceView().displayableManagerByClassName(
            "vtkMRMLSegmentationsDisplayableManager2D"
        )

        # Get segment ids at RAS position
        segment_ids = vtk.vtkStringArray()
        seg_disp_manager.GetVisibleSegmentsForPosition(
            ras_pos, self._node.segmentation.GetDisplayNode(), segment_ids
        )

        # Set the first segment found as the selected segment
        if segment_ids.GetNumberOfValues() > 0:
            self._segment_editor.self().editor.setCurrentSegmentID(segment_ids.GetValue(0))

    def _handle_segment_added(self, caller, event):
        """Add a corresponding object to the Scene Graph. If necessary, also ask the user for the object class."""
        LOAD_STATE_MANAGER.selected_patient_scene_graph_modified = True
        obj_id = LOAD_STATE_MANAGER.scene_graph.get_next_available_bounding_box_id()

        # Note: we cannot rely on the added segment being last,
        #       because an undo operation will put the new segment directly at its original index
        existing_ids = self._node.segmentation.GetSegmentation().GetSegmentIDs()
        for i, segment_id in enumerate(existing_ids):
            if segment_id not in self._segment_id_to_sg_object:
                break
        else:
            print("handle_segment_added: No matching added segment found")
            return

        # Compute list of possible obj classes
        obj_classes = []
        for obj_class in LOAD_STATE_MANAGER.knowledge_graph.classes:
            if obj_class.is_ignored:
                # Class is ignored so ignore it
                continue
            if obj_class.is_unique \
                    and len(LOAD_STATE_MANAGER.scene_graph.bounding_boxes_by_class_id.get(obj_class.id, [])) > 0:
                # Class is unique and already in the scene graph
                continue
            obj_classes.append(obj_class)

        if len(obj_classes) == 0:
            # No possible object class, so we just delete the segment
            obj_class_id = 0
        elif len(obj_classes) == 1:
            # Only one possible object class, so we use it and don't ask the user
            obj_class_id = obj_classes[0].id
        else:
            # Create the input dialog
            input_dialog = qt.QInputDialog()
            input_dialog.setOptions(qt.QInputDialog.UseListViewForComboBoxItems)
            # Set the selection mode to ComboBox
            mapping = {f"{obj_class.id}: {obj_class.name}": obj_class.id for obj_class in obj_classes}
            names = list(mapping.keys())
            selected_value = input_dialog.getItem(
                slicer.util.mainWindow(),  # Parent widget
                "Select Object Class",  # Window Title
                "Select the object class for the new segment:",  # Label text inside the window
                names,  # List of strings
                -1,  # Default index
                False  # Editable (False = user must pick from the list)
            )

            if selected_value != "":
                obj_class_id = mapping[selected_value]
            else:
                obj_class_id = 0

        if obj_class_id == 0:
            # No possible object class, so we just delete the segment.
            # That's at least what we would want to do, but the segment editor has a bug where it will try to access it.
            # So we have to trigger the deletion event with a slight delay.
            self._timer.timeout.connect(lambda: self._node.segmentation.RemoveSegment(segment_id))
            self._timer.start()
            return

        # Add the object to the Scene Graph
        knowledge_graph = LOAD_STATE_MANAGER.knowledge_graph
        obj = BoundingBox(
            obj_class_id,
            obj_id,
            BoundingBox.default_name(knowledge_graph, obj_class_id, obj_id),
            (tuple(), tuple())  # Temporary bounding box as it will get recomputed before saving
        )
        obj.init_attributes_keypoints(
            knowledge_graph.get_object_class_by_id(obj_class_id),
            len(LOAD_STATE_MANAGER.scene_graph.object_labelmap.shape),
            knowledge_graph.obj_common
        )
        LOAD_STATE_MANAGER.scene_graph.add_bounding_box(obj)
        # Update mappings
        self._segment_id_to_sg_object[segment_id] = obj
        self._sg_object_to_segment_id[obj.id] = segment_id
        # Update segment name
        self._node.segmentation.GetSegmentation().GetSegment(segment_id).SetName(obj.name)
        # Signal to other widgets
        LOAD_STATE_MANAGER.sg_object_added(obj)

        # We have to call this again, because the color update for the new object's widget need to be triggered again
        # The original call to _handle_segment_modified will not work because it triggered too early
        self._handle_segment_modified(None, None)

    def _handle_segment_modified(self, caller, event):
        """Mark the patient as modified and update names if necessary."""
        old_colors = LOAD_STATE_MANAGER.colors
        new_colors = {}
        signal_color_change = False

        LOAD_STATE_MANAGER.selected_patient_scene_graph_modified = True
        # Check all segments for name changes
        for segment_id, sg_obj in self._segment_id_to_sg_object.items():
            segment = self._node.segmentation.GetSegmentation().GetSegment(segment_id)
            segment_name = segment.GetName()
            if sg_obj.name != segment_name:
                sg_obj.name = segment_name
                LOAD_STATE_MANAGER.name_changed.emit(sg_obj)
            # Color update
            rgb = segment.GetColor()
            r, g, b = [int(round(c * 255)) for c in rgb]
            hex_string = f"#{r:02x}{g:02x}{b:02x}".upper()
            new_colors[sg_obj.id] = hex_string
            signal_color_change |= old_colors.get(sg_obj.id) != hex_string

        LOAD_STATE_MANAGER.colors = new_colors
        # Only emit signal if there was a change
        if signal_color_change:
            LOAD_STATE_MANAGER.update_color.emit()

    def _handle_segment_removed(self, caller, event):
        """Remove the corresponding object from the Scene Graph."""
        remaining_ids = self._node.segmentation.GetSegmentation().GetSegmentIDs()
        for segment_id in self._segment_id_to_sg_object:
            if segment_id not in remaining_ids:
                break
        else:
            print("handle_segment_removed: No matching deleted segment found")
            return

        LOAD_STATE_MANAGER.selected_patient_scene_graph_modified = True
        obj: BoundingBox = self._segment_id_to_sg_object[segment_id]
        # Update mappings
        del self._segment_id_to_sg_object[segment_id]
        del self._sg_object_to_segment_id[obj.id]
        # Signal to other widgets
        LOAD_STATE_MANAGER.safe_remove_object_from_scene_graph(obj)

    def external_segment_delete(self, obj_id: int):
        """Remove the corresponding object from the segmentation and the Scene Graph (triggers all relevant stuff)."""
        if self._node is None or self._node.segmentation is None:
            print("external_segment_delete: No segmentation loaded")
            return

        segment_id = self._sg_object_to_segment_id.get(obj_id)
        if segment_id is None:
            print(f"external_segment_delete: No segment found for object id {obj_id}")
            return

        # Save state for undo from editor
        self._segment_editor.self().editor.saveStateForUndo()
        self._node.segmentation.GetSegmentation().RemoveSegment(segment_id)
        self._segment_editor.updateWidgetFromMRML()

    def _toggle_ai_prediction(self):
        """Toggle the AI prediction."""
        if self._node is None:
            print("toggle_ai_prediction: No parameter node loaded")
            return
        if self._node.ai_segmentation is not None:
            self._node.ai_segmentation.GetDisplayNode().SetVisibility(bool(LOAD_STATE_MANAGER.show_ai_prediction))

    def _handle_name_change(self, obj: BoundingBox):
        """Update the name of the corresponding segment."""
        if self._node is None:
            print("handle_name_changed: No parameter node loaded")
            return
        if self._node.segmentation is None:
            print("handle_name_changed: No segmentation loaded")
            return

        segment_id = self._sg_object_to_segment_id[obj.id]
        segment = self._node.segmentation.GetSegmentation().GetSegment(segment_id)
        if segment.GetName() != obj.name:
            segment.SetName(obj.name)

    def _handle_center_views_on_object(self, obj: BoundingBox):
        """Center the views on the object."""
        if self._node is None:
            print("handle_center_views_on_object: No parameter node loaded")
            return
        if self._node.segmentation is None:
            print("handle_center_views_on_object: No segmentation loaded")
            return

        segment_id = self._sg_object_to_segment_id[obj.id]
        position = self._node.segmentation.GetSegmentCenterRAS(segment_id)
        for sliceNode in slicer.util.getNodesByClass('vtkMRMLSliceNode'):
            sliceNode.JumpSliceByCentering(*position)
