import os
import pathlib


YOLO_WEIGHT_PATH = os.path.join(pathlib.Path(__file__).parent.absolute(), "yolov3.weights")
YOLO_CONFIG_PATH = os.path.join(pathlib.Path(__file__).parent.absolute(), 'yolov3.cfg')
YOLO_COCO_PATH = os.path.join(pathlib.Path(__file__).parent.absolute(), 'coco.names')

