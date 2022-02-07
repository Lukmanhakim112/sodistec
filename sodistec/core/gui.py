import numpy as np

try:
    from cv2 import cv2
except ImportError:
    import cv2

from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import (
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
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

    def _setting_group(self):
        group_box = QGroupBox()
        layout = QGridLayout()

        font = QFont('Arial', 13)

        distance_label = QLabel("Jarak Minimal: ")
        self.distance_input = QLineEdit()
        distance_sbutt = QPushButton("Simpan Jarak")

        distance_sbutt.clicked.connect(self._set_max_distance)

        layout.addWidget(distance_label, 0, 0)
        layout.addWidget(self.distance_input, 0, 2, 1, 2)
        layout.addWidget(distance_sbutt, 0, 4)

        group_box.setLayout(layout)
        group_box.setTitle("Setting")
        group_box.setFont(font)

        return group_box

    def _info_group(self):
        group_box = QGroupBox()
        layout = QHBoxLayout()

        font = QFont('Arial', 13)

        self.person_label = QLabel("Total Orang: 0")
        self.violation_label = QLabel("Total Pelanggar: 0")
        self.min_dist_label = QLabel(f"Jarak Minimal: {config.MIN_DISTANCE}")

        layout.addWidget(self.person_label)
        layout.addWidget(self.violation_label)
        layout.addWidget(self.min_dist_label)

        group_box.setLayout(layout)
        group_box.setTitle("Informasi")
        group_box.setFont(font)

        return group_box

    def _feed_group(self):
        group_box = QGroupBox()
        layout = QGridLayout()

        self.opencv_box = QLabel(self)
        self.opencv_box.resize(1280, 720)
        self.opencv_box.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.opencv_box, 0, 0)

        group_box.setLayout(layout)
        group_box.setTitle("Informasi")

        return group_box

    @pyqtSlot()
    def _set_max_distance(self) -> None:
        try:
            value = int(self.distance_input.text())
        except Exception:
            value = -1

        if value < 0:
            return

        config.MIN_DISTANCE = value
        self.min_dist_label.setText(f"Jarak Minimal: {config.MIN_DISTANCE}")

    def _set_textbox(self) -> None:
        self.textbox = QLineEdit(self)

    def _init_ui(self) -> None:
        self.setWindowTitle(self.title)
        self.setGeometry(40, 60, 1280, 720)

        self.grid.addWidget(self._setting_group(), 0, 0)
        self.grid.addWidget(self._feed_group(), 1, 0)
        self.grid.addWidget(self._info_group(), 2, 0)

        self.video_input = DetectPerson(config.CAMERA_URL)
        # connect image from opencv to qt
        self.video_input.change_pixmap_signal.connect(self.update_image)

        # some parameters from openct, passed to qt
        self.video_input.total_people_signal.connect(self._update_total_person)
        self.video_input.total_serious_violations_signal.connect(self._update_total_serious_violations)

        self.video_input.start()

    @pyqtSlot()
    def _min_distance(self) -> None:
        self._set_min_distance(int(self.min_distance_textbox.text()))

    @pyqtSlot(int)
    def _update_total_person(self, total_people) -> None:
        self.person_label.setText(f'Total Orang: {total_people}')

    @pyqtSlot(int)
    def _update_total_serious_violations(self, total_serious_violations) -> None:
        self.violation_label.setText(f'Total Pelanggar: {total_serious_violations}')

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

