import sys

from sodistec.core.gui import WindowApp

from PyQt5.QtWidgets import QApplication

import qdarktheme

def main(argv):

    app = QApplication(argv)
    ex = WindowApp()
    app.setStyleSheet(qdarktheme.load_stylesheet("light"))
    ex.showMaximized()

    sys.exit(app.exec())



if __name__ == '__main__':
    main(sys.argv)
