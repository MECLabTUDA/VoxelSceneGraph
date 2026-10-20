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

from PyQt6.QtCore import pyqtSignal, Qt, QSize
from PyQt6.QtGui import QPixmap, QMouseEvent, QWheelEvent
from PyQt6.QtWidgets import QLabel, QSizePolicy


class QPixmapLabelWithMouseTracking(QLabel):
    """
    QLabel containing a QPixmap and also tracking th cursor position within the image_array.
    If always_track is False, move event will only be emitted if a mouse button is pressed.
    keep_mouse_in_bounds can be used to move the cursor to the opposite end of the pixmap,
    if it is getting out of bounds while a mouse button is pressed.
    The label will expand to take as much space as possible (even when the underlying pixmap is smaller).
    """

    cursor_moved = pyqtSignal(int, int)
    # Triggered only when keep_mouse_in_bounds == True
    # May not work correctly if the pixmap is not displayed entirely
    cursor_was_repositioned = pyqtSignal(int, int)
    cursor_pressed = pyqtSignal(Qt.MouseButton, int, int)
    cursor_released = pyqtSignal(Qt.MouseButton, int, int)
    wheel_scrolled = pyqtSignal(int)

    def __init__(self, pixmap: QPixmap, always_track: bool = False, keep_mouse_in_bounds: bool = False):
        """
        :param always_track: Whether the cursor coordinates are also tracked when no button is being held
        :param keep_mouse_in_bounds: Whether the mouse is moved to the opposite side of the pixmap when leaving it
                                     while a button is pressed
        """
        super().__init__()
        self.setPixmap(pixmap)
        self.always_track = always_track
        self.keep_mouse_in_bounds = keep_mouse_in_bounds
        self.init_ui()

    def init_ui(self):
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        self.setMouseTracking(self.always_track)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def sizeHint(self):
        # In combination with the right size policy, makes the label take as much place as possible
        return QSize(1 << 30, 1 << 30)

    def _clean_cursor_coordinates(self, x: int, y: int):
        """
        Given raw coordinates, returns them without the label size offset and also clipped to the bounds.
        Coordinates out of bounds happen if a mouse button is pressed inside the pixmap and
        the cursor moves out of it.
        """
        # TODO handle when the cursor goes out of screen or the window is split over multiple screens
        # Remove offset caused by the QLabel having a different size than the QPixmap
        x -= (self.width() - self.pixmap().width()) // 2
        y -= (self.height() - self.pixmap().height()) // 2
        # Clip to bounds
        clipped_x = min(self.pixmap().width() - 1, max(0, x))
        clipped_y = min(self.pixmap().height() - 1, max(0, y))
        return x, y, clipped_x, clipped_y

    def mouseMoveEvent(self, ev: QMouseEvent):
        """
        Handler for mouse move events.
        If not keep_mouse_in_bounds, simply clips the coordinates and emits them.
        Otherwise, if the cursor is out of bounds:
        - determine which side of the pixmap the cursor is getting through
        - emit that the cursor will be repositioned
        - move it to the opposite side (which triggers a move event)
        """
        # If the cursor is pressed within the image and then moves outside
        # You get positions outside the image_array
        raw_x, raw_y = ev.pos().x(), ev.pos().y()
        x, y, clipped_x, clipped_y = self._clean_cursor_coordinates(raw_x, raw_y)

        if self.keep_mouse_in_bounds and (clipped_x != x or clipped_y != y):
            # Does not require to keep track of whether a button is currently pressed,
            # as the widget will only produce events that are out of bound if a button is pressed
            # Cursor being out of bounds is detected when the clean coordinates don't match the clipped ones
            cursor_x, cursor_y = self.cursor().pos().x(), self.cursor().pos().y()

            # Move the cursor the correct way, only works if the entire width of the pixmap displayed
            # Also update coordinates that we want to emit:
            # => don't use the clipped ones as if any clipping occurred, they are slightly moved
            #    and the new coordinated will be in bounds anyway
            if x < 0:
                cursor_x += self.pixmap().width() - 1
                x += self.pixmap().width() - 1
            elif x > self.pixmap().width() - 1:
                cursor_x -= self.pixmap().width() - 1
                x -= self.pixmap().width() - 1

            # Move the cursor the correct way, only works if the entire height of the pixmap displayed
            if y < 0:
                cursor_y += self.pixmap().height() - 1
                y += self.pixmap().height() - 1
            elif y > self.pixmap().height() - 1:
                cursor_y -= self.pixmap().height() - 1
                y -= self.pixmap().height() - 1

            # Emit that we're moving the cursor so that other algorithm can update their cursor coordinates
            self.cursor_was_repositioned.emit(x, y)
            # Will then trigger a normal move signal
            self.cursor().setPos(cursor_x, cursor_y)
        else:
            self.cursor_moved.emit(clipped_x, clipped_y)

    def mousePressEvent(self, ev: QMouseEvent):
        self._signal_mouse_button_press(ev, True)

    def mouseReleaseEvent(self, ev: QMouseEvent):
        self._signal_mouse_button_press(ev, False)

    def _signal_mouse_button_press(self, ev: QMouseEvent, pressed: bool):
        """Handler for mouse buttons being pressed or released."""
        _, _, x, y = self._clean_cursor_coordinates(ev.pos().x(), ev.pos().y())
        if pressed:
            self.cursor_pressed.emit(ev.button(), x, y)
        else:
            self.cursor_released.emit(ev.button(), x, y)

    def wheelEvent(self, ev: QWheelEvent):
        """Handler for the mouse being scrolled."""
        self.wheel_scrolled.emit(ev.angleDelta().y())
