import os

# File path to this folder
THIS_PATH = os.path.dirname(os.path.realpath(__file__))

YOLO_WEIGHT_PATH = os.path.join(THIS_PATH, "yolov3.weights")
YOLO_CONFIG_PATH = os.path.join(THIS_PATH, 'yolov3.cfg')
YOLO_COCO_PATH = os.path.join(THIS_PATH, 'coco.names')

YOLO4_MINI_WEIGHT_PATH = os.path.join(THIS_PATH, "yolov4-tiny.weights")
YOLO4_MINI_CONFIG_PATH = os.path.join(THIS_PATH, 'yolov4-tiny.cfg')
