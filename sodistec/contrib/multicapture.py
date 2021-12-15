import queue, threading

from cv2 import cv2

class CaptureThread:
    def __init__(self, input_name) -> None:
        # Select input capture i.e: camera
        self.capture = cv2.VideoCapture(input_name)

        # define empty queue
        self.queue = queue.Queue()

        # Create and start threading daemon
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _reader(self) -> None:
        while True:
            (ret, frame) = self.capture.read()

            # exit when ret not supplied
            if not ret:
                break

            if not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass

            self.queue.put(frame)

    def read(self):
        return self.queue.get()

