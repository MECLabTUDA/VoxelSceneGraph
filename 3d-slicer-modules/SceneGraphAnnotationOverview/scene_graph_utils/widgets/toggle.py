import qt
from PythonQt.QtCore import Qt, QSize
from PythonQt.QtGui import QPainter, QColor


class ResponsiveToggle(qt.QCheckBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Allows the widget to be resized by layouts
        self.setSizePolicy(qt.QSizePolicy.Policy.Preferred, qt.QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def sizeHint(self):
        """Defines the 'natural' size based on standard scaling."""
        # Typically, a toggle is ~2x as wide as it is high.
        # 24px is a standard height for many desktop buttons.
        return QSize(48, 24)

    def hitButton(self, pos):
        return self.rect.contains(pos)

    def paintEvent(self, event):
        # Initialize QPainter and begin painting
        painter = QPainter(self)

        try:
            # Enable anti-aliasing for smoother drawing
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Get widget dimensions and values
            is_on = self.isChecked()
            h = self.height
            w = self.width
            track_color = QColor("#4cd964") if is_on else QColor("#bdc3c7")

            # Draw Track
            painter.setBrush(track_color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(0, 0, w, h, h / 2, h / 2)

            # Draw Handle
            margin = h * 0.1
            handle_size = h - (margin * 2)
            x_pos = w - handle_size - margin if is_on else margin

            painter.setBrush(QColor("white"))
            painter.drawEllipse(x_pos, margin, handle_size, handle_size)

        finally:
            # Ensure the painter is properly ended to avoid errors
            painter.end()
