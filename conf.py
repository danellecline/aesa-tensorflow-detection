TOP_DIR = '/Users/dcline/Dropbox/GitHub/aesa-tensorflow-detection/'
#OPTIMIZE_BOX = True
# Categories to refine the bounding box on
OPTIMIZED_CATEGORIES = ['ENYPNIASTESEXIMIA', 'PENIAGONE', 'AMPERIMA']
OPTIMIZED_GROUPS = ['HOLOTHUROIDEA', "ARTHOPODA', 'ASTEROIDEA"]
COLLECTION = 'M56'
OPTIMIZE_BOX = False
COLLECTION = 'M56_960x540' #'M56_1920x1080'
DATA_DIR = '{0}/data/'.format(TOP_DIR)
ANNOTATION_DIR = '{0}/{1}/Annotations/'.format(DATA_DIR, COLLECTION)
PNG_DIR = '{0}/{1}/PNGImages'.format(DATA_DIR, COLLECTION)
ANNOTATION_FILE = '{0}/annotations/M56_Annotations_v13.csv'.format(DATA_DIR)
GROUP_FILE = '{0}/annotations/hierarchy_group_k5.csv'.format(DATA_DIR)
LABEL_PATH_PATH = '{0}/aesa_k5_label_map.pbtxt'.format(TOP_DIR)
TILE_DIR = '/Volumes/ScratchDrive/AESA/M56_tiles/raw/'
TILE_PNG_DIR = '/Volumes/ScratchDrive/AESA/M56_images_960_540'

#Alternative file prefix to use for calculating the associated frame the annotation is from, e.g. M56_10441297_%d.jpg'")
FILE_FORMAT = 'M56_10441297_'

# xml constants
DATABASE='Unknown'
DEPTH='3'
SEGMENTED='0'
POSE = 'Unspecified'
DIFFICULT = '0'

# Annotation constants
PAD_PERCENTAGE = 25
MIN_CROP = 75
TARGET_TILE_WIDTH = 600 #960
TARGET_TILE_HEIGHT = 600 #540

SHOW_ANNOTATION = False
