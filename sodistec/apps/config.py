from sodistec.contrib.yolo import yolo
from sodistec.contrib.caffe import caffe

# Load COCO class label
LABEL_PATH = yolo.YOLO_COCO_PATH
LABELS = open(LABEL_PATH).read().strip().split("\n")

# YOLO Config
YOLO_WEIGHT_PATH = yolo.YOLO_WEIGHT_PATH
YOLO_CONFIG_PATH = yolo.YOLO_CONFIG_PATH

# Caffe Config
CAFFE_WEIGHT_PATH = caffe.CAFFE_WEIGHT_PATH
CAFFE_PROTEXT_PATH = caffe.CAFFE_PROTEXT_PATH

# Define MAX and MIN (in pixels) distance beetwen 2 people
MIN_DISTANCE = 50
MAX_DISTANCE = 80

# Define minimum probability to filter weak detection
# with the threashold when applying non-maxima suppression
MIN_CONF = 0.2
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
CAMERA_URL = None

# Email setting
EMAIL_ALERT = False
EMAIL_ADDRESS = None
