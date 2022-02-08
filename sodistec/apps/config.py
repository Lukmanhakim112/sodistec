from typing import Union

from sodistec.contrib.yolo import yolo

# Load COCO class label
with open(yolo.YOLO_COCO_PATH) as f:
    LABELS = f.read().strip().split("\n")

# YOLO Config
YOLO_WEIGHT_PATH = yolo.YOLO_WEIGHT_PATH
YOLO_CONFIG_PATH = yolo.YOLO_CONFIG_PATH

# Define MIN (in pixels) distance beetwen 2 people
MIN_DISTANCE: int = 300

# distance beetwen person by it's distance
# to the camera
MIN_RADIUS: int = 35 # 20 

# MAX_DISTANCE = 160
PLAY_BUZZER: bool = False

# Define minimum probability to filter weak detection
# with the threashold when applying non-maxima suppression
MIN_CONF: float = 0.25
NMS_THRESH: float = 0.3

# Use GPU for the computations
USE_GPU: bool = True

# Show counter for the people
SHOW_PEOPLE_COUNTER: bool = True

# Vialotaions limit
THERESHOLD: int = 15

# Use threading
USE_THREADING: bool = True

# Set IP Camera url
# set 0 for using a webcam
CAMERA_URL: Union[int, str] = "rtsp://admin:labiot2018@192.168.1.55/live/ch00_01"
CAMERA_URL_2: Union[int, str] = "rtsp://admin:labiot2018@192.168.1.54/live/ch00_01"

