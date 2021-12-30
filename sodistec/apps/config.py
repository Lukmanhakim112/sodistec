from sodistec.contrib.yolo import yolo

# Load COCO class label
with open(yolo.YOLO_COCO_PATH) as f:
    LABELS = f.read().strip().split("\n")

# YOLO Config
YOLO_WEIGHT_PATH = yolo.YOLO_WEIGHT_PATH
YOLO_CONFIG_PATH = yolo.YOLO_CONFIG_PATH

# Define MAX and MIN (in pixels) distance beetwen 2 people
MIN_DISTANCE = 500

# distance beetwen person by it's distance
# to the camera
MIN_DISTANCE_X = 5 

# MAX_DISTANCE = 160

# Define minimum probability to filter weak detection
# with the threashold when applying non-maxima suppression
MIN_CONF = 0.25
NMS_THRESH = 0.3

# Use GPU for the computations
USE_GPU = True

# Show counter for the people
SHOW_PEOPLE_COUNTER = True

# Vialotaions limit
THERESHOLD = 15

# Use threading
USE_THREADING = True

# Set IP Camera url
# set 0 for using a webcam
CAMERA_URL = 0
