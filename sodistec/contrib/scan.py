import socket
import typing

import nmap

from PyQt5.QtWidgets import QDialog, QWidget

class PortScanner(QDialog):
    def __init__(self, parent: typing.Optional[QWidget] = ...) -> None:
        super(PortScanner, self).__init__(parent)

        host = socket.gethostname()
        ip = socket.gethostbyname(host)
        ip = ip.split(".")
        ip[3] = "0"
        ip.append("/24")

        nm = nmap.PortScanner()
        print(nm.scan(hosts="".join(ip)))
