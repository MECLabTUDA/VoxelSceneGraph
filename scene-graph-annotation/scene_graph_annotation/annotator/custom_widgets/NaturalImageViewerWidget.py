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

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon, QCursor, QPixmap
from PyQt6.QtWidgets import QWidget, QLabel, QSizePolicy, QVBoxLayout, QHBoxLayout, QFrame, QToolButton, \
    QButtonGroup, QSpinBox, QApplication

from scene_graph_annotation.scene import SceneGraph
from scene_graph_annotation.ui_utils import QPixmapLabelWithMouseTracking
from scene_graph_annotation.utils.asset_paths import cursor_icon, move_icon, zoom_in_icon, zoom_out_icon, reload_icon
from scene_graph_annotation.utils.image_utils import rgba_array_to_qpixmap, combine_rgba_images
from scene_graph_annotation.utils.timers import RESIZE_TIMEOUT
from .UIManager import UIManager
from ...utils.ArrayView import NaturalImageArrayView


# Note: copy/paste of RadiologyImageViewerWidget, but with bits removed and the overlay rendering adapted
# noinspection DuplicatedCode
class NaturalImageViewerWidget(QWidget):
    """
    Widget used to display a 2D slice of a 3D volume with annotation.
    In click/select mode, clicking a mask will select the corresponding object as subject or object
    using respectively the left or right click. If the corresponding object is already selected, it will be unselected.
    On resize, the content of the QPixmap will be updated (optimized with a single shot timer).
    Contains:
    - a toolbox row (click, move | zoom in, zoom out, label "Zoom: {}%" | Label+opacity | reset view)
    - the label with the image
    The cursor will also change when hovering the pixmap, depending on the selected mode.
    If in select mode, the tooltip will also be the name of the object under the cursor.

    Note: since we might get multiple signals one after the other (each triggering a pixmap update),
          we use a single shot timer with a very short (reasonable) timer to avoid multiple computation.
    """

    # Timeout to avoid updating the canvas multiple times when e.g. splitting objects
    # This sort of action will cause multiple canvas update
    # But we don't want the delay to seem noticeable
    CANVAS_UPDATE_TIMEOUT = 10

    def __init__(
            self,
            image: NaturalImageArrayView,
            scene_graph: SceneGraph,
            ui_manager: UIManager,
    ):
        super().__init__()
        self._image = image
        self._scene_graph = scene_graph
        self._ui_manager = ui_manager

        self._object_color_darkening = 0.8  # Factor by which an object annotation mask is multiplied when selected
        self._zoom_incr = 0.25
        self._default_opacity = 50  # Set default opacity as in 3D Slicer

        # Scaling of mouse movement to value change
        self._scroll_dy = 20  # 20 px for 1 slice
        self._windowing_d = 10  # 10 px for 1 slice

        # Keep track of mouse coordinates for scroll and windowing
        self._last_cursor_x = 0
        self._last_cursor_y = 0
        self._button_pressed: Qt.MouseButton | None = None  # If not None, button being pressed

        # Cursor constants
        # Note: these cannot be class attributes because Qt is not yet initialized when this class is loaded
        # Click mode
        self._CLICK_MODE_CURSOR_DEFAULT = QCursor(Qt.CursorShape.ArrowCursor)  # No obj under cursor
        self._CLICK_MODE_CURSOR_CLICKABLE = QCursor(Qt.CursorShape.PointingHandCursor)  # Obj under cursor
        # Move mode
        self._MOVE_MODE_CURSOR_DEFAULT = QCursor(Qt.CursorShape.OpenHandCursor)  # When not holding the l-mouse button
        self._MOVE_MODE_CURSOR_MOVING = QCursor(Qt.CursorShape.ClosedHandCursor)  # When holding the left mouse button

        # Top row
        self._click_button = QToolButton()
        self._move_button = QToolButton()
        self._zoom_in_button = QToolButton()
        self._zoom_out_button = QToolButton()
        self._zoom_level_label = QLabel()
        self._opacity_spinbox = QSpinBox()
        self._reset_button = QToolButton()
        self._toolbox_button_group = QButtonGroup(self)

        # Icons
        self._zoom_in_icon = QIcon(zoom_in_icon.as_posix())
        self._zoom_out_icon = QIcon(zoom_out_icon.as_posix())

        # Image label
        # Always track for dynamic cursor
        # Dummy target shape here as this widget will be resized after initialization anyway
        self._image_label = QPixmapLabelWithMouseTracking(QPixmap(), always_track=True, keep_mouse_in_bounds=True)

        # Timer fo resize
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(lambda: self._update_pixmap_content(update_overlay=True))

        # Since we don't select different slices, the overlay should remain the same
        # (unless an object is added or removed or subject/object changed)
        self._rgba_overlay: np.ndarray | None = None

        # Connect external signals, add a small timeout to avoid updating the canvas too often
        ui_manager.subject_changed.connect(lambda: self._timer.start(self.CANVAS_UPDATE_TIMEOUT))
        ui_manager.object_changed.connect(lambda: self._timer.start(self.CANVAS_UPDATE_TIMEOUT))
        ui_manager.sg_object_added.connect(lambda obj: self._timer.start(self.CANVAS_UPDATE_TIMEOUT))
        ui_manager.sg_object_removed.connect(lambda obj: self._timer.start(self.CANVAS_UPDATE_TIMEOUT))
        ui_manager.show_object.connect(self._show_object)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Set button states, do this before anything else to avoid clicked trigger
        for button in [self._click_button, self._move_button]:
            button.setCheckable(True)
            self._toolbox_button_group.addButton(button)
        self._click_button.setChecked(True)

        # ==============================================================================================================
        # Tool box
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(frame)
        layout.setContentsMargins(0, 0, 0, 0)

        toolbox_widget = QWidget()
        toolbox_layout = QHBoxLayout()
        toolbox_widget.setLayout(toolbox_layout)
        toolbox_layout.setContentsMargins(0, 0, 0, 0)
        toolbox_widget.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)

        toolbox_layout.addStretch()
        self._click_button.setIcon(QIcon(cursor_icon.as_posix()))
        toolbox_layout.addWidget(self._click_button)
        # Cursor change
        self._click_button.clicked.connect(lambda: self._image_label.setCursor(self._CLICK_MODE_CURSOR_DEFAULT))
        self._click_button.setToolTip("Object selection")

        toolbox_layout.addSpacing(5)
        self._move_button.setIcon(QIcon(move_icon.as_posix()))
        toolbox_layout.addWidget(self._move_button)
        # Cursor change
        self._move_button.clicked.connect(lambda: self._image_label.setCursor(self._MOVE_MODE_CURSOR_DEFAULT))
        self._move_button.setToolTip("Panning")

        toolbox_layout.addSpacing(5)
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.VLine)
        toolbox_layout.addWidget(frame)
        toolbox_layout.addSpacing(5)

        self._zoom_in_button.setIcon(self._zoom_in_icon)
        toolbox_layout.addWidget(self._zoom_in_button)
        self._zoom_in_button.clicked.connect(
            lambda: self._handle_zoom(self._zoom_incr) or self._update_pixmap_content()
        )
        self._zoom_in_button.setToolTip("Zoom in")

        toolbox_layout.addSpacing(5)
        self._zoom_out_button.setIcon(self._zoom_out_icon)
        toolbox_layout.addWidget(self._zoom_out_button)
        self._zoom_out_button.clicked.connect(
            lambda: self._handle_zoom(-self._zoom_incr) or self._update_pixmap_content()
        )
        self._zoom_out_button.setToolTip("Zoom out")

        toolbox_layout.addSpacing(5)
        toolbox_layout.addWidget(self._zoom_level_label)

        toolbox_layout.addSpacing(5)
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.VLine)
        toolbox_layout.addWidget(frame)
        toolbox_layout.addSpacing(5)

        label = QLabel("Mask opacity (%):")
        label.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        toolbox_layout.addWidget(label)
        self._opacity_spinbox.setRange(0, 100)
        self._opacity_spinbox.setValue(self._default_opacity)
        self._opacity_spinbox.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self._opacity_spinbox.setSingleStep(5)
        toolbox_layout.addWidget(self._opacity_spinbox)
        self._opacity_spinbox.valueChanged.connect(self._update_pixmap_content)

        toolbox_layout.addSpacing(5)
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.VLine)
        toolbox_layout.addWidget(frame)
        toolbox_layout.addSpacing(5)

        self._reset_button.setIcon(QIcon(reload_icon.as_posix()))
        toolbox_layout.addWidget(self._reset_button)
        self._reset_button.clicked.connect(self._reset_view)
        self._reset_button.setToolTip("Reset view")
        toolbox_layout.addStretch()

        layout.addWidget(toolbox_widget)
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(frame)

        # ==============================================================================================================
        # Image label
        self._image_label.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        layout.addWidget(self._image_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self._image_label.wheel_scrolled.connect(self._wheel_handler)
        self._image_label.cursor_pressed.connect(self._handle_click)
        self._image_label.cursor_released.connect(self._handle_release)
        self._image_label.cursor_moved.connect(self._handle_move)
        self._image_label.cursor_was_repositioned.connect(self._handle_cursor_reposition)

        # Update and display mask
        self._reset_view()

    def _update_pixmap_content(self, update_overlay: bool = False):
        """
        Updates the pixmap content:
        - take a slice out of the image
        - take a slice out of the annotation if the opacity is not 100%
        - if the subject or the object is displayed, their mask is made slightly darker and increase opacity
        - combine the two rgba arrays
        - compute the view from the combined slice
        """
        opacity = self._opacity_spinbox.value() * 255 // 100
        # Width is fine, height is slightly too much
        self._image.target_size = self._image_label.width(), self._image_label.height()

        # Update or compute object overlay
        if self._rgba_overlay is None or update_overlay:
            rgb_overlay = self._scene_graph.object_overlay
            rgba_overlay = np.dstack((rgb_overlay, np.zeros(rgb_overlay.shape[:2], dtype=np.uint8)))
            all_obj_idx_mask = self._scene_graph.object_hitboxes > 0
            rgba_overlay[all_obj_idx_mask, -1] = opacity

            # Make color of selected subject/object darker
            # Note: since this gets computed again on subject/object change, we can do it here (and only once)
            for obj in [self._ui_manager.subject, self._ui_manager.object]:
                if obj is None:
                    continue
                obj_idx_mask = self._scene_graph.object_hitboxes == obj.id
                rgba_overlay[obj_idx_mask, :-1] = \
                    rgba_overlay[obj_idx_mask, :-1] * self._object_color_darkening
                rgba_overlay[obj_idx_mask, -1] = (255 + opacity) // 2  # Increase by half that is missing

            self._rgba_overlay = rgba_overlay
        else:
            rgba_overlay = self._rgba_overlay

        rgba_image = self._image.array
        # Setting opacity, contours and darkening when object selected
        if opacity > 0:
            # Combining two views is much faster than combining 2 images and then taking a view
            # (At least for images larger than the space available for display)
            image_view = self._image.get_view(rgba_image)
            overlay_view = self._image.get_view(rgba_overlay)
            final_view = combine_rgba_images(image_view, overlay_view)
        else:
            # Finally compute the view and set it as the pixmap content
            final_view = self._image.get_view(rgba_image)

        self._image_label.setPixmap(rgba_array_to_qpixmap(final_view))

    def _wheel_handler(self, wheel_dy: int):
        """Mousewheel callback for changing slices or zooming."""

        if QApplication.keyboardModifiers() & Qt.KeyboardModifier.ControlModifier:
            # Ctrl pressed => zoom
            self._handle_zoom((1 if wheel_dy > 0 else -1) * self._zoom_incr)
            self._update_pixmap_content()

    def _handle_click(self, button: Qt.MouseButton, x: int, y: int):
        """
        Callback for handling clicks:
        - In click mode: updates the cursor icon when hovering annotations (when not completely transparent)
        - In move mode: updates the cursor icon when panning
        Also tracks whether the left button was pressed and the mouse coordinates for this event.
        """

        if self._button_pressed is not None:
            # Do nothing as another button is already pressed and we don't want to interfere
            return

        # Click mode
        if self._click_button.isChecked():
            # Instead of up-scaling the hitbox array, we can rather map rescaled array coordinates
            # back to original coordinates
            # This removes the need to compute a view of the hitbox array
            try:
                orig_coordinates = self._image.view_coordinates_to_orig_array(x, y)
                obj_id = self._scene_graph.object_hitboxes[orig_coordinates]
            except IndexError:
                obj_id = 0

            # Object found and not completely transparent
            if obj_id > 0 and self._opacity_spinbox.value() > 0:
                if button == Qt.MouseButton.LeftButton:
                    # Set subject if different else unselect
                    if self._ui_manager.subject != self._scene_graph.get_bounding_box_by_id(obj_id):
                        self._ui_manager.subject = self._scene_graph.get_bounding_box_by_id(obj_id)
                    else:
                        self._ui_manager.subject = None
                elif button == Qt.MouseButton.RightButton:
                    # Set object if different else unselect
                    if self._ui_manager.object != self._scene_graph.get_bounding_box_by_id(obj_id):
                        self._ui_manager.object = self._scene_graph.get_bounding_box_by_id(obj_id)
                    else:
                        self._ui_manager.object = None
                # Note: no need to call this explicitly as changing the subject/object already calls this
                # self._update_pixmap_content()
                # Note: it's important to return here, as the rest of the code handles cursor movement
                return

        # The code below handles mouse movement behaviours

        # Move button
        # Note: we also allow to move using the click mode if no object has been clicked
        if ((self._move_button.isChecked() or self._click_button.isChecked()) and button == Qt.MouseButton.LeftButton) \
                or button == Qt.MouseButton.MiddleButton:
            # Set cursor to hand
            self._image_label.setCursor(self._MOVE_MODE_CURSOR_MOVING)

        # Update button being pressed, but only if there's not another one already pressed
        self._button_pressed = button

        # Update last values for later
        self._last_cursor_x = x
        self._last_cursor_y = y

    def _handle_zoom(self, zoom_incr: float):
        """Callback for changing the zoom level."""
        self._image.zoom_factor += zoom_incr
        self._zoom_level_label.setText(f"Zoom: {int(self._image.zoom_factor * 100)}%")

    def _handle_release(self, button: Qt.MouseButton, _: int, __: int):
        """Callback for handling mouse release. Mostly updates cursor icon and the left button pressed flag."""

        if button != self._button_pressed:
            # Do nothing as another button was initially pressed and we don't want to interfere
            return

        if self._click_button.isChecked():
            self._image_label.setCursor(self._CLICK_MODE_CURSOR_DEFAULT)
        if self._move_button.isChecked():
            self._image_label.setCursor(self._MOVE_MODE_CURSOR_DEFAULT)

        self._button_pressed = None

    def _handle_cursor_reposition(self, x: int, y: int):
        """
        Callback used to update the internal tracking of the cursor.
        Also updates the cursor icon if necessary.
        Note: this callback is necessary for when the cursor is moved programmatically,
              e.g. loop back from one edge of the image to the other.
        """
        # Update last values for later, but don't trigger anything that is cursor position delta based (e.g. scroll)
        self._last_cursor_x = x
        self._last_cursor_y = y
        # But still update the cursor icon
        self._click_mode_hovering_mode_update(x, y)

    def _handle_move(self, x: int, y: int):
        """
        Callback for handling mouse movements:
        - In click mode:
          - when left button down and no object clicked: move mode
          - otherwise: update the cursor and tooltip
        - In move mode: updates the anchor for the panning
        - In scroll mode: changes the selected slice and keeps track of partial (not integer) slice changes.
        - In windowing mode: changes window center and width and keeps track of partial value changes.
        - In all modes + middle button: move mode
        """

        if self._button_pressed is None:
            # Hovering mode
            self._handle_cursor_reposition(x, y)
            return

        is_left_mouse_button = self._button_pressed == Qt.MouseButton.LeftButton

        # Panning
        if is_left_mouse_button or self._button_pressed == Qt.MouseButton.MiddleButton:
            # Note: since self._button_pressed is not None only in move mode,
            #       we know that when we clicked, we haven't clicked on an object
            self._image.move_anchor(self._last_cursor_x - x, self._last_cursor_y - y)
            self._update_pixmap_content()

        self._last_cursor_x = x
        self._last_cursor_y = y

    def _click_mode_hovering_mode_update(self, x: int, y: int):
        """Method used to update the cursor icon depending on what it is hovering and in the click mode."""
        # Cursor visualization
        # https://het.as.utexas.edu/HET/Software/PyQt/qcursor.html

        if self._button_pressed is not None:
            # Button pressed => not in hovering mode
            return

        self.setToolTip("")
        if self._click_button.isChecked():
            # Dynamic cursor update so that the normal cursor becomes a hand when hovering a mask

            # Instead of up-scaling the hitbox array, we can rather map rescaled array coordinates
            # back to original coordinates
            # This removes the need to compute a view of the hitbox array
            try:
                orig_coordinates = self._image.view_coordinates_to_orig_array(x, y)
                obj_id = self._scene_graph.object_hitboxes[orig_coordinates]
            except IndexError:
                obj_id = 0

            # Object is found and not completely transparent
            if obj_id != 0 and self._opacity_spinbox.value() > 0:
                self._image_label.setCursor(self._CLICK_MODE_CURSOR_CLICKABLE)
                self.setToolTip(self._scene_graph.get_bounding_box_by_id(obj_id).name)
            else:
                self._image_label.setCursor(self._CLICK_MODE_CURSOR_DEFAULT)

    def _reset_view(self):
        """Sets the view to the original default one."""
        # Reset zoom level to 100% and update the label
        self._handle_zoom(1. - self._image.zoom_factor)
        self._image.reset_anchor()
        self._opacity_spinbox.setValue(self._default_opacity)
        self._update_pixmap_content()

    def resizeEvent(self, a0):
        # We need to resize the pixmap to fit the new size of the label
        # self._update_pixmap_content()
        self._timer.start(RESIZE_TIMEOUT)

    def _show_object(self, object_id: int):
        """Callback used to change the selected slice to show an object to the user."""
        bbox = self._scene_graph.get_bounding_box_by_id(object_id).bounding_box
        center = tuple(x1 + (x2 - x1) / 2 for x1, x2 in zip(*bbox))
        self._image.center_anchor_on(*center)
        self._update_pixmap_content()
