import typing

import serial

from PyQt5.QtCore import QThread, pyqtSignal, QObject


class TemperatureReader(QThread):
    temperature = pyqtSignal(str)

    def __init__(self, port_number: str, parent: typing.Optional[QObject] = ...) -> None:
        super(TemperatureReader, self).__init__(parent)

        self.serial = serial.Serial(
            port=port_number,
            baudrate=9600,
            parity=serial.PARITY_ODD,
            stopbits=serial.STOPBITS_TWO,
            bytesize=serial.SEVENBITS
        )
        self.scanning = True

    def run(self) -> None:

        while self.scanning:
            try:
                data = self.serial.readline().decode('utf-8')
                temp = float(data.strip('\n\r'))

                self.temperature.emit(f"Suhu: {temp} °C")
            except Exception:
                pass

