import numpy as np

try:
    from cv2 import cv2
except ImportError:
    import cv2

from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import (
    QGridLayout, QLabel, QLineEdit,
    QPushButton, QWidget
)

from sodistec.apps import config
from sodistec.core.detection import DetectPerson

class WindowApp(QWidget):

    def __init__(self) -> None:
        super().__init__()

        self.WIDGET_HEIGHT = 25

        self.title = "Sodistec"

        self.grid = QGridLayout(self)
        self.grid.setRowMinimumHeight(0, self.WIDGET_HEIGHT)

        font = QFont()
        font.setPointSize(11)
        self.qfont = font

        self._init_ui()

    def _add_to_grid(self,widget: QWidget, row: int, column: int, row_span: int = 1, col_span: int = 1) -> None:
        # add widget to desired location 
        self.grid.addWidget(widget, row, column, row_span, col_span)

    def _set_min_distance(self, value: int) -> None:
        config.MIN_DISTANCE = value

    def _set_max_distance(self, value: int) -> None:
        config.MAX_DISTANCE = value

    def _set_textbox(self) -> None:
        self.textbox = QLineEdit(self)

    def _init_ui(self) -> None:
        self.setWindowTitle(self.title)
        self.setGeometry(40, 60, 1280, 720)

        self.opencv_box = QLabel(self)
        self.opencv_box.resize(1280, 720)
        self.opencv_box.setAlignment(Qt.AlignCenter)
        
        b_min_distance = QPushButton("Simpan jarak minimal", self)
        b_min_distance.setFont(self.qfont)
        label_max_distance = QLabel("Jarak minimal: ", self) 
        label_max_distance.setFont(self.qfont)
        self._add_to_grid(label_max_distance, 0, 0)
        self.min_distance_textbox = QLineEdit(str(config.MIN_DISTANCE))
        self.min_distance_textbox.setFont(self.qfont)
        self._add_to_grid(self.min_distance_textbox, 0, 1)
        self._add_to_grid(b_min_distance, 0, 2)

        b_max_distance = QPushButton("Simpan jarak maksimal", self)
        b_max_distance.setFont(self.qfont)
        label_min_distance = QLabel("Jarak maksimal: ", self)
        label_min_distance.setFont(self.qfont)
        self._add_to_grid(label_min_distance, 1, 0)
        self.max_distance_textbox = QLineEdit(str(config.MAX_DISTANCE))
        self.max_distance_textbox.setFont(self.qfont)
        self._add_to_grid(self.max_distance_textbox, 1, 1)
        self._add_to_grid(b_max_distance, 1, 2)

        self._add_to_grid(self.opencv_box, 2, 0, 1, 3)

        self.total_violations = QLabel("Total Pelanggaran: 10")
        self.total_violations.setFont(self.qfont)
        self._add_to_grid(self.total_violations, 3, 0)

        self.serious_violations = QLabel("Total Pelanggaran Serius: 10")
        self.serious_violations.setFont(self.qfont)
        self._add_to_grid(self.serious_violations, 4, 0)

        self.total_people = QLabel("Total Orang: 10")
        self.total_people.setFont(self.qfont)
        self._add_to_grid(self.total_people, 3, 1)
        
        self.maximum_distance = QLabel(f'Jarak Maksimal: {config.MAX_DISTANCE}')
        self.maximum_distance.setFont(self.qfont)
        self._add_to_grid(self.maximum_distance, 4, 1)

        b_min_distance.clicked.connect(self._min_distance)
        b_max_distance.clicked.connect(self._max_distance)

        self.video_input = DetectPerson(0)
        # connect image from opencv to qt
        self.video_input.change_pixmap_signal.connect(self.update_image)

        # some parameters from openct, passed to qt
        self.video_input.total_violations_signal.connect(self._update_total_violations)
        self.video_input.total_people_signal.connect(self._update_total_person)
        self.video_input.total_serious_violations_signal.connect(self._update_total_serious_violations)

        self.video_input.start()

    @pyqtSlot()
    def _min_distance(self) -> None:
        self._set_min_distance(int(self.min_distance_textbox.text()))

    @pyqtSlot()
    def _max_distance(self) -> None:
        self._set_max_distance(int(self.max_distance_textbox.text()))
        self.maximum_distance.setText(f'Jarak Maksimal: {self.max_distance_textbox.text()}')

    @pyqtSlot(int)
    def _update_total_person(self, total_people) -> None:
        self.total_people.setText(f'Total Orang: {total_people}')

    @pyqtSlot(int)
    def _update_total_violations(self, total_violations) -> None:
        self.total_violations.setText(f'Total Pelanggaran: {total_violations}')

    @pyqtSlot(int)
    def _update_total_serious_violations(self, total_serious_violations) -> None:
        self.serious_violations.setText(f'Total Pelanggaran Serius: {total_serious_violations}')

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img) -> None:
        qt_image = self.convert_cv_qt(cv_img)
        self.opencv_box.setPixmap(qt_image)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        convert_to_Qt_format = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(960, 540)
        return QPixmap.fromImage(p)

