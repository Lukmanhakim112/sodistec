import numpy as np

try:
    from cv2 import cv2
except ImportError:
    import cv2

from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import (
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QWidget, QVBoxLayout
)

from sodistec.apps import config
from sodistec.core.detection import DetectPerson

class WindowApp(QWidget):

    def __init__(self) -> None:
        super().__init__()

        self.WIDGET_HEIGHT = 25

        self.title = "Sodistec"

        # Display buffer
        self.display_feed = {}
        self.cameras = {}

        # Statistic buffer
        self.people_counter = {}
        self.violation_counter = {}

        self.grid = QGridLayout(self)
        self.grid.setRowMinimumHeight(0, self.WIDGET_HEIGHT)

        font = QFont()
        font.setPointSize(11)
        self.qfont = font

        self._init_ui()

    def _add_info(self, index: int):
        group_box = QGroupBox()

        layout = QHBoxLayout()
        font = QFont('Arial', 13)

        self.people_counter[index] = QLabel("Total Orang: 0")
        self.violation_counter[index] = QLabel("Total Pelanggar: 0")

        group_box.setLayout(layout)
        group_box.setTitle(f"Informasi Camera {index + 1}")
        group_box.setFont(font)

        layout.addWidget(self.people_counter[index])
        layout.addWidget(self.violation_counter[index])

        return group_box

    def _add_video_feed(self, index: int):
        group_box = QGroupBox()
        layout = QVBoxLayout()
        font = QFont('Arial', 13)

        self.display_feed[f"display_{index}"] = QLabel(self)
        self.display_feed[f"display_{index}"].resize(1280, 720)
        self.display_feed[f"display_{index}"].setAlignment(Qt.AlignCenter)

        group_box.setLayout(layout)
        group_box.setFont(font)
        group_box.setTitle(f"Camera {index + 1}")

        layout.addWidget(self.display_feed[f"display_{index}"])
        layout.addWidget(self._add_info(index))
        layout.setStretch(0, 1)

        return group_box

    def _feed_group(self):
        group_box = QGroupBox()
        layout = QHBoxLayout()

        for index, camera in enumerate(config.CAMERAS_URL):
            self.cameras[f"camera_{index}"] = DetectPerson(camera, index)
            self.cameras[f"camera_{index}"].change_pixmap_signal.connect(self._update_image)
            self.cameras[f"camera_{index}"].total_people_signal.connect(self._update_total_person)
            self.cameras[f"camera_{index}"].total_serious_violations_signal.connect(self._update_total_serious_violations)

            layout.addWidget(self._add_video_feed(index))

        group_box.setLayout(layout)
        return group_box

    def _setting_group(self):
        group_box = QGroupBox()
        layout = QGridLayout()

        font = QFont('Arial', 13)

        distance_label = QLabel("Jarak Minimal: ")
        self.distance_input = QLineEdit()
        distance_sbutt = QPushButton("Simpan Jarak")

        self.min_dist_label = QLabel(f"Jarak Minimal: {config.MIN_DISTANCE}")

        distance_sbutt.clicked.connect(self._set_max_distance)

        layout.addWidget(distance_label, 0, 0)
        layout.addWidget(self.distance_input, 0, 2, 1, 2)
        layout.addWidget(distance_sbutt, 0, 4)
        layout.addWidget(self.min_dist_label, 1, 0, 1, 5)

        group_box.setLayout(layout)
        group_box.setTitle("Setting")
        group_box.setFont(font)

        return group_box

    @pyqtSlot()
    def _set_max_distance(self) -> None:
        try:
            value = int(self.distance_input.text())
        except Exception:
            return

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

        self.grid.setRowStretch(1, 2)

        for index, camera in enumerate(config.CAMERAS_URL):
            self.cameras[f"camera_{index}"].start()

    @pyqtSlot()
    def _min_distance(self) -> None:
        self._set_min_distance(int(self.min_distance_textbox.text()))

    @pyqtSlot(int, int)
    def _update_total_person(self, total_people, camera_id) -> None:
        self.people_counter[camera_id].setText(f'Total Orang: {total_people}')

    @pyqtSlot(int, int)
    def _update_total_serious_violations(self, total_serious_violations, camera_id) -> None:
        self.violation_counter[camera_id].setText(f'Total Pelanggar: {total_serious_violations}')

    @pyqtSlot(np.ndarray, int)
    def _update_image(self, cv_img, camera_id) -> None:
        qt_image = self.convert_cv_qt(cv_img)
        self.display_feed[f"display_{camera_id}"].setPixmap(qt_image)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        convert_to_Qt_format = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(960, 540)
        return QPixmap.fromImage(p)

