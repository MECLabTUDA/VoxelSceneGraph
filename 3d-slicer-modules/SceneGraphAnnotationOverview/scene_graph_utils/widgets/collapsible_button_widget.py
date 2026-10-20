import qt
import slicer


class CollapsibleButtonWidget(qt.QWidget):
    """Class that exists because some qMRML widgets are not really QObjects."""
    def __init__(self):
        super().__init__()

        layout = qt.QFormLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        self.button = slicer.qMRMLCollapsibleButton()
        layout.addWidget(self.button)

    def setButtonLayout(self, layout):
        if self.button.layout():
            self.button.layout().deleteLater()
        self.button.setLayout(layout)

    def setText(self, text: str):
        self.button.setText(text)
