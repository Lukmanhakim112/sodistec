import sys

from sodistec.core.detection import DetectPerson
from sodistec.core.gui import WindowApp

from PyQt5.QtWidgets import QApplication

def main(argv):
    #  p1 = DetectPerson(".\\videos\\test.mp4")
    #  #  p1 = DetectPerson(1)
    #  p1.run()

    app = QApplication(argv)
    ex = WindowApp()
    ex.show()

    sys.exit(app.exec())



if __name__ == '__main__':
    main(sys.argv)
