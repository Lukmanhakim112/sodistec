import sys

from PyQt5.QtWidgets import (
    QDialog, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget
)

from sodistec.apps import config

def to_int(word: str):
    try:
        return int(word)
    except ValueError:
        return word

class SetCamera(QDialog):
    def __init__(self, parent: QWidget) -> None:
        super(SetCamera, self).__init__(parent)

        self.parent: QWidget = parent;

        self.setWindowTitle("Set Camera")
        self.resize(640, 480)

        self.camera_text = QTextEdit(self)
        with open('camera_url_list') as f:
            self.camera_text.setText("".join(list(map(str, f.read()))))

        save_button = QPushButton("Simpan URL Kamera", self)

        save_button.clicked.connect(self.set_cameras_url)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Silahkan Memasukan URL IP Camera: ", self))
        layout.addWidget(QLabel("Pisahkan setiap URL dengan garis baru (enter)", self))
        layout.addWidget(self.camera_text)
        layout.addWidget(save_button)

        self.setLayout(layout)

    def closeEvent(self, event) -> None:
        sys.exit()

    def set_cameras_url(self):
        text = self.camera_text.toPlainText().strip()

        config.CAMERAS_URL = list(map(to_int, text.split("\n")))

        with open('camera_url_list', 'w') as f:
            f.write(text)

        self.hide()


