import threading
from multiprocessing import Process, Pool

try:
    from cv2 import cv2
except ImportError:
    import cv2

class CaptureThread:
    def __init__(self, input_name) -> None:

        self.capture = cv2.VideoCapture(input_name)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 3)

        (self.grabed, self.frame) = self.capture.read()

        self.stopped = False

    def start(self):
        # Create and start threading
        t = threading.Thread(target=self._update)
        t.daemon = True
        t.start()

        # Or process?
        # TODO: try using multiprocessing
        #  p = Pool()
        #  p.imap(self._update, self.frame)
        
        return self

    def _stop(self) -> None:
        self.stopped = True

    def _update(self):
        while True:

            if self.stopped:
                break
            (self.grabed, self.frame) = self.capture.read()

    def read(self):
        return self.frame


