TOP_DIR = '/Users/dcline/Dropbox/GitHub/aesa-tensorflow-detection/'
DATA_DIR = '{0}/data/'.format(TOP_DIR)

# Categories to refine the bounding box on
OPTIMIZED_CATEGORIES = ['ENYPNIASTESEXIMIA', 'PENIAGONE', 'AMPERIMA']
OPTIMIZED_GROUPS = ['HOLOTHUROIDEA', "ARTHOPODA', 'ASTEROIDEA"]
OPTIMIZE_BOX = False

TARGET_TILE_WIDTH = 600
TARGET_TILE_HEIGHT = 600
LABEL_PATH_PATH = '{0}/data/aesa_group_map.pbtxt'.format(TOP_DIR)
COLLECTION_M535455 =  'M535455_{0}x{1}_by_group'.format(TARGET_TILE_WIDTH, TARGET_TILE_HEIGHT)
COLLECTION_M56 =  'M56_{0}x{1}_by_group'.format(TARGET_TILE_WIDTH, TARGET_TILE_HEIGHT)


COLLECTION = COLLECTION_M535455
ANNOTATION_DIR = '{0}/{1}/Annotations/'.format(DATA_DIR, COLLECTION)
PNG_DIR = '{0}/{1}/PNGImages'.format(DATA_DIR, COLLECTION)
ANNOTATION_FILE = '{0}/annotations/M535455_Annotations_v10.csv'.format(DATA_DIR)
GROUP_FILE = None
CLEAN_FILE = '{0}/annotations/M535455_crops_JMD.csv'.format(DATA_DIR)
TILE_DIR = '/Volumes/ScratchDrive/AESA/M535455_tiles/'
TILE_PNG_DIR = '/Volumes/ScratchDrive/AESA/M535455_10_1'
FILE_FORMAT = '%s.jpg'

'''
COLLECTION = COLLECTION_M56
ANNOTATION_DIR = '{0}/{1}/Annotations/'.format(DATA_DIR, COLLECTION)
PNG_DIR = '{0}/{1}/PNGImages'.format(DATA_DIR, COLLECTION)
ANNOTATION_FILE = '{0}/annotations/M56_Annotations_v13.csv'.format(DATA_DIR)
GROUP_FILE = '{0}/annotations/M56_categories_and_groups_v13.csv'.format(DATA_DIR)
HIERARCHY_FILE = '{0}/annotations/hierarchy_group_k5.csv'.format(DATA_DIR)
CLEAN_FILE = None
TILE_DIR = '/Volumes/ScratchDrive/AESA/M56_tiles/raw/'
TILE_PNG_DIR = '/Volumes/ScratchDrive/AESA/M56_10_1'
#Alternative file prefix to use for calculating the associated frame the annotation is from, e.g. M56_10441297_%d.jpg'")
FILE_FORMAT = 'M56_10441297_%s.jpg'''''

# xml constants
DATABASE='Unknown'
DEPTH='3'
SEGMENTED='0'
POSE = 'Unspecified'
DIFFICULT = '0'

# Annotation constants
PAD_PERCENTAGE = 25
MIN_CROP = 75
SHOW_ANNOTATION = False
