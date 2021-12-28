from threading import Thread
import time

import numpy as np
from playsound import playsound

try:
    from cv2 import cv2
except ImportError:
    import cv2

from PyQt5.QtCore import QThread, pyqtSignal

from scipy.spatial import distance as dist

from sodistec.apps import config
from sodistec.contrib.yolo import yolo
from sodistec.contrib.multicapture import CaptureThread

class DetectPerson(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    total_violations_signal = pyqtSignal(int)
    total_serious_violations_signal = pyqtSignal(int)
    total_people_signal = pyqtSignal(int)
    safe_distance_signal = pyqtSignal(int)

    def __init__(self, video_input, detect: str = "person", 
                 use_gpu: bool = config.USE_GPU,
                 use_threading: bool = config.USE_THREADING,
                 parent = None
                ) -> None:
        super(DetectPerson, self).__init__(parent)

        self.detect = detect
        self.model = cv2.dnn.readNetFromDarknet(
            yolo.YOLO4_MINI_CONFIG_PATH, yolo.YOLO4_MINI_WEIGHT_PATH
        )

        layer = self.model.getLayerNames()
        self.layer = [layer[i - 1] for i in self.model.getUnconnectedOutLayers()]


        if use_gpu:
            self._use_gpu()

        self._set_video_capture(video_input, use_threading)

    def _set_video_capture(self, video_input, use_threading) -> None:
        print("[INFO] Setup video feed...")
        if use_threading:
            self.video_capture = CaptureThread(video_input).start()
        else:
            self.video_capture = cv2.VideoCapture(video_input)

    def _use_gpu(self) -> None:
        print("[INFO] Searching for compatible NVIDIA GPU...")
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def _play_buzzer(self) -> None:
        playsound("./sodistec/core/buzzer.wav")

    def _detect_people(self, frame, person_index: int = 0) -> list:
        # grab the dimensions of the frame and  initialize the list of
        # results
        (H, W) = frame.shape[:2]
        results = []

        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
        self.model.setInput(blob)
        layerOutputs = self.model.forward(self.layer)

        # initialize our lists of detected bounding boxes, centroids, and
        # confidences, respectively
        boxes = []
        centroids = []
        confidences = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # confidence is met
                if class_id == person_index and confidence > config.MIN_CONF:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates,
                    # centroids, and confidences
                    boxes.append([x, y, int(width), int(height)])
                    centroids.append((centerX, centerY))
                    confidences.append(float(confidence))

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, config.MIN_CONF, config.NMS_THRESH)
        self.total_people_signal.emit(len(idxs))
        #  if config.SHOW_PEOPLE_COUNTER:
        #      human_count = "Total Orang: {}".format(len(idxs))
        #      cv2.putText(frame, human_count, (470, frame.shape[0] - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 0), 2)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # update our results list to consist of the person
                # prediction probability, bounding box coordinates,
                # and the centroid
                results.append((confidences[i], (x, y, x + w, y + h), centroids[i]))

        return results

    def run(self) -> None:
        while True:
            # read the next frame from the file
            if config.USE_THREADING:
                frame = self.video_capture.read()
                time.sleep(0.01)
            else:
                (grabbed, frame) = self.video_capture.read()
                # if the frame was not grabbed, then we have reached the end of the stream
                if not grabbed:
                    break

            # resize the frame and then detect people (and only people) in it
            #  frame = imutils.resize(frame, width=640)
            frame  = cv2.resize(frame, (960, 540), cv2.INTER_LINEAR)
            results = self._detect_people(frame, person_index=config.LABELS.index("person"))

            # initialize the set of indexes that violate the max/min social distance limits
            serious = set()
            abnormal = set()

            # ensure there are *at least* two people detections (required in
            # order to compute our pairwise distance maps)
            if len(results) >= 2:
                # extract all centroids from the results and compute the
                # Euclidean distances between all pairs of the centroids
                centroids = np.array([r[2] for r in results])
                data = dist.cdist(centroids, centroids, metric="euclidean")

                # loop over the upper triangular of the distance matrix
                for i in range(0, data.shape[0]):
                    for j in range(i + 1, data.shape[1]):
                        # check to see if the distance between any two
                        # centroid pairs is less than the configured number of pixels
                        if data[i, j] < config.MIN_DISTANCE:
                            serious.add(i)
                            serious.add(j)
                        elif (data[i, j] < config.MAX_DISTANCE):
                            Thread(target=self._play_buzzer).start() # PLAY SOUND!!
                            abnormal.add(i)
                            abnormal.add(j)

            # loop over the results 
            for (i, (_, bbox, centroid)) in enumerate(results):
                # extract the bounding box and centroid coordinates, then
                # initialize the color of the annotation
                (startX, startY, endX, endY) = bbox
                (cX, cY) = centroid
                color = (0, 255, 0)

                # if the index pair exists within the violation/abnormal sets, then update the color
                if i in serious:
                    color = (0, 0, 255)
                elif i in abnormal:
                    color = (0, 255, 255) #orange = (0, 165, 255)

                # draw (1) a bounding box around the person and (2) the
                # centroid coordinates of the person,
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.circle(frame, (cX, cY), 5, color, 2)

            #  # draw some of the parameters
            #  Safe_Distance = "Jarak aman: > {}px".format(config.MAX_DISTANCE)
            #  cv2.putText(frame, Safe_Distance, (470, frame.shape[0] - 25),
            #      cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)
            #  self.safe_distance_signal.emit(config.MAX_DISTANCE)

            #  Threshold = "Maksimal pelanggaran: {}".format(config.THERESHOLD)
            #  cv2.putText(frame, Threshold, (470, frame.shape[0] - 50),
            #      cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)

            #  # draw the total number of social distancing violations on the output frame
            #  text = "Total pelanggaran serius: {}".format(len(serious))
            #  cv2.putText(frame, text, (10, frame.shape[0] - 55),
            #      cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)
            self.total_serious_violations_signal.emit(len(serious))

            #  text1 = "Total pelanggaran: {}".format(len(abnormal))
            #  cv2.putText(frame, text1, (10, frame.shape[0] - 25),
            #      cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)
            self.total_violations_signal.emit(len(abnormal))

            #  cv2.imshow("Testing", frame)
            self.change_pixmap_signal.emit(frame)

