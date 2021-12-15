import numpy as np
from cv2 import cv2

import imutils
#  from imutils.video import FPS

from scipy.spatial import distance as dist

from sodistec.apps import config
from sodistec.contrib.multicapture import CaptureThread

class DetectPerson:
    def __init__(self, video_input, detect: str = "person" ,use_gpu: bool = config.USE_GPU, use_threading: bool = config.USE_THREADING):

        self.detect = detect
        self.model = cv2.dnn.readNetFromDarknet(
            config.YOLO_CONFIG_PATH, config.YOLO_WEIGHT_PATH
        )

        layer = self.model.getLayerNames()
        self.layer = [layer[i - 1] for i in self.model.getUnconnectedOutLayers()]

        if use_gpu:
            self._use_gpu()

        self._set_video_capture(video_input, use_threading)

    def _set_video_capture(self, video_input, use_threading):
        print("[INFO] Capturing video feed...")
        if use_threading:
            self.video_capture = CaptureThread(video_input)
        else:
            self.video_capture = cv2.VideoCapture(video_input)

    def _use_gpu(self):
        print("[INFO] Searching for compatible NVIDIA GPU...")
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def _detect_people(self, frame, person_index: int = 0):
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

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # filter detections by (1) ensuring that the object
                # detected was a person and (2) that the minimum
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
        if config.SHOW_PEOPLE_COUNTER:
            human_count = "Human count: {}".format(len(idxs))
            cv2.putText(frame, human_count, (470, frame.shape[0] - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 0), 2)

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
                r = (confidences[i], (x, y, x + w, y + h), centroids[i])
                results.append(r)

        return results

    def run(self):
        while True:
            # read the next frame from the file
            if config.USE_THREADING:
                frame = self.video_capture.read()

            else:
                (grabbed, frame) = self.video_capture.read()
                # if the frame was not grabbed, then we have reached the end of the stream
                if not grabbed:
                    break

            # resize the frame and then detect people (and only people) in it
            frame = imutils.resize(frame, width=700)
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
                D = dist.cdist(centroids, centroids, metric="euclidean")

                # loop over the upper triangular of the distance matrix
                for i in range(0, D.shape[0]):
                    for j in range(i + 1, D.shape[1]):
                        # check to see if the distance between any two
                        # centroid pairs is less than the configured number of pixels
                        if D[i, j] < config.MIN_DISTANCE:
                            # update our violation set with the indexes of the centroid pairs
                            serious.add(i)
                            serious.add(j)
                        # update our abnormal set if the centroid distance is below max distance limit
                        if (D[i, j] < config.MAX_DISTANCE) and not serious:
                            abnormal.add(i)
                            abnormal.add(j)

            # loop over the results 
            for (i, (prob, bbox, centroid)) in enumerate(results):
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

                
                # draw some of the parameters
                Safe_Distance = "Safe distance: >{} px".format(config.MAX_DISTANCE)
                cv2.putText(frame, Safe_Distance, (470, frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)
                Threshold = "Threshold limit: {}".format(config.THERESHOLD)
                cv2.putText(frame, Threshold, (470, frame.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)

                # draw the total number of social distancing violations on the output frame
                text = "Total serious violations: {}".format(len(serious))
                cv2.putText(frame, text, (10, frame.shape[0] - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)

                text1 = "Total abnormal violations: {}".format(len(abnormal))
                cv2.putText(frame, text1, (10, frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)


                cv2.imshow("Testing", frame)
                key = cv2.waitKey(1) & 0xFF

                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    break




