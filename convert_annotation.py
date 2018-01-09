#!/usr/bin/env python
from utils import ensure_dir

__author__ = 'Danelle Cline'
__copyright__ = '2017'
__license__ = 'GPL v3'
__contact__ = 'dcline at mbari.org'
__doc__ = '''

Reads in AESA annotation file and converts annotations into a Tensorflow record for object detection tests
Converts the tiles into a grid of tiles along the longest dimension and adjusts the annotation to map 
to this new coordinate system

@var __date__: Date of last svn commit
@undocumented: __doc__ parser
@status: production
@license: GPL
'''

import math
import numpy
import pandas as pd
from xml.dom.minidom import parseString
import shutil
from collections import defaultdict
from collections import namedtuple
import argparse
import cv2
import hashlib
import conf
import io
import os
import sys
import tensorflow as tf
from image_utils import find_object, get_dims

sys.path.append(os.path.join(os.path.dirname(__file__), 'tensorflow_models', 'research'))

from PIL import Image
from dicttoxml import dicttoxml
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       image_subdirectory='imgs'):
  """Convert csv derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding fields for a single image
    dataset_directory: Path to root directory holding dataset
    label_map_dict: A map from string label names to integers ids.
    image_subdirectory: String specifying subdirectory within the
    dataset directory holding the actual image data.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  img_path = os.path.join(data['folder'], image_subdirectory, data['filename'])
  full_path = os.path.join(dataset_directory, img_path)
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_png = fid.read()
  encoded_png_io = io.BytesIO(encoded_png)
  image = Image.open(encoded_png_io)
  if image.format != 'PNG':
    raise ValueError('Image format not PNG')
  key = hashlib.sha256(encoded_png).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  num_objs = 0

  for obj in data['object']:
    num_objs += 1
    difficult = bool(int(obj['difficult']))
    difficult_obj.append(int(difficult))
    xmin.append(float(obj['bndbox']['xmin']) / width)
    ymin.append(float(obj['bndbox']['ymin']) / height)
    xmax.append(float(obj['bndbox']['xmax']) / width)
    ymax.append(float(obj['bndbox']['ymax']) / height)
    classes_text.append(obj['name'].encode('utf8'))
    classes.append(label_map_dict[obj['name']])
    truncated.append(int(obj['truncated']))
    poses.append(obj['pose'].encode('utf8'))

  example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': dataset_util.int64_feature(height),
    'image/width': dataset_util.int64_feature(width),
    'image/filename': dataset_util.bytes_feature(
      data['filename'].encode('utf8')),
    'image/source_id': dataset_util.bytes_feature(
      data['filename'].encode('utf8')),
    'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
    'image/encoded': dataset_util.bytes_feature(encoded_png),
    'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
    'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
    'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
    'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
    'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
    'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
    'image/object/class/label': dataset_util.int64_list_feature(classes),
    'image/object/truncated': dataset_util.int64_list_feature(truncated),
    'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return example, num_objs


def convert_annotation(raw_file, img_dir, annotation, image_height, image_width, tile_height, tile_width):
  '''
   Convert annotation to dictionary
  :param raw_file:  path to file with tile
  :param img_dir: directory tiled images are in
  :param img_frame_dir: directory to store tile grid
  :param annotation: annotation tuple
  :return:
  '''
  if "Length" in annotation.mtype and not math.isnan(annotation.measurement):
    crop_pixels = 2 * int(float(annotation.measurement))
  else:
    crop_pixels = 500

  head, tail = os.path.split(raw_file)
  stem = tail.split('.')[0]

  values_x = annotation.centerx
  values_y = annotation.centery

  bins_x = numpy.arange(start=0, stop=image_width - tile_width, step=tile_width - 1)
  bins_y = numpy.arange(start=0, stop=image_height - tile_height, step=tile_height - 1)

  posx = numpy.digitize(values_x, bins_x)
  posy = numpy.digitize(values_y, bins_y)

  index = (posy - 1) * int(image_width / tile_width) + (posx - 1)

  #print('Index: {0} center x {1} centery {2}'.format(index, values_x, values_y))

  image_file = '{0}_{1:02}.png'.format(stem, index)

  img = cv2.imread('{0}/{1}'.format(img_dir, image_file))
  center_x = values_x - (posx - 1) * tile_width
  center_y = values_y - (posy - 1) * tile_height
  tlx = max(0, int(center_x - crop_pixels / 2))
  tly = max(0, int(center_y - crop_pixels / 2))
  if a.mtype == 'Length':
    brx = min(tile_width, int(tlx + crop_pixels))
    bry = min(tile_height, int(tly + crop_pixels))
  else:
    brx = min(tile_width, int(tlx + crop_pixels / 2))
    bry = min(tile_height, int(tly + crop_pixels / 2))

  '''crop_img = img[tly:tly + int(crop_pixels), tlx:tlx + int(crop_pixels)]
  gray_image = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
  display_annotation(annotation, brx, bry, img, tlx, tly)

  ret2, th = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
  kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
  erode = cv2.erode(th, kernel, iterations=1)
  clean = cv2.dilate(erode, kernel2, iterations=1) 
  
  # first try Otsu
  cv2.imshow('Otsu', clean)
  found, x, y, w, h = find_object(clean, crop_img)

  if 'Cnidaria' not in a.category and 'Ophiur' not in a.category:
    if not found:
      # Next try threshold in increments of 2
      for thresh in range(3, 13, 2):
        th = cv2.adaptiveThreshold(th, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                   cv2.THRESH_BINARY, thresh, 2)
        cv2.imshow('Thresh {0}'.format(thresh), th)
        found, x, y, w, h = find_object(th, crop_img)
        if found:
          break;

  if found:
    # adjust the actual bounding box with new localized bounding box coordinates
    tlx += x;
    tly += y;
    brx = tlx + w;
    bry = tly + h;

  cv2.destroyAllWindows()
  display_annotation(annotation, brx, bry, img, tlx, tly)'''
  obj = {}
  obj['name'] = a.category
  obj['difficult'] = conf.DIFFICULT
  obj['truncated'] = 0
  obj['pose'] = conf.POSE
  obj['bndbox'] = {}
  obj['bndbox']['xmin'] = tlx
  obj['bndbox']['ymin'] = tly
  obj['bndbox']['xmax'] = brx
  obj['bndbox']['ymax'] = bry
  return obj, image_file


def display_annotation(annotation, brx, bry, img, tlx, tly):
  cv2.rectangle(img, (tlx, tly), (brx, bry), (0, 255, 0), 3)
  font = cv2.FONT_HERSHEY_SIMPLEX
  cv2.putText(img, annotation.category, (tlx, tly), font, 2, (255, 255, 255), 2)
  cv2.imshow("Annotation", img)


def correct(annotation):
  '''
  Correct annotation categories
  :param annotation:
  :return: corrected annotation
  '''
  if 'OphiuroideaDiskD' in annotation.category or 'OphiuroideaR' in annotation.category:
    annotation.category = 'Ophiuroidea'

  if 'Tunicate2' in annotation.category:
    annotation.category = 'Tunicata2'

  if 'Polycheata1' in annotation.category:
    annotation.category = 'Polychaete1'
    # annotation.group = 'Polychaeta'

  return annotation


if __name__ == '__main__':

  ensure_dir(conf.DATA_DIR)
  failed_file = open(os.path.join(conf.DATA_DIR, 'failed_crops.txt'), 'w')
  png_dir = os.path.join(conf.PNG_DIR)
  ensure_dir(png_dir)
  label_map_dict = label_map_util.get_label_map_dict(conf.LABEL_PATH_PATH)
  aesa_annotation = namedtuple("Annotation", ["category", "centerx", "centery", "mtype", "measurement", "index"])

  try:
    print('Parsing ' + conf.ANNOTATION_FILE)
    df = pd.read_csv(conf.ANNOTATION_FILE, sep=',')
    print('Parsing ' + conf.GROUP_FILE)
    df_group = pd.read_csv(conf.GROUP_FILE, sep=',')
    group_map = {}
    for index, row in df_group.iterrows():
      if numpy.isnan(float(row['coarse_class'])):
        print('Category {0} does not have a group'.format(row['category']))
      else:
        print('Grouping category {0} as GROUP{1}'.format(row['category'], int(row['coarse_class'])))
        group_map[row['category']] = 'GROUP{:d}'.format(int(row['coarse_class']))

    width = None
    height = None
    TARGET_TILE_WIDTH = 960
    TARGET_TILE_HEIGHT = 540
    n_tilesh = None
    n_tilesw = None
    actual_width = None
    actual_height = None
    frame_dict = {}
    annotation_dict = {}
    ntest = 0
    ttl_objs = 0

    for index, row in sorted(df.iterrows()):
      try:
        # if index > 1:
        #  break;
        f = row['FileName']
        if conf.FILE_FORMAT:
          filename = os.path.join(conf.TILE_DIR, '{0}{1}.jpg'.format(conf.FILE_FORMAT, f))
        else:
          filename = os.path.join(conf.TILE_DIR, f)

        # filename = '{0}/data/{1}'.format(os.getcwd(), 'M56_10441297_12987348614060.jpg' )
        if not os.path.exists(filename):
          failed_file.writelines(filename)
          continue

        # get image height and width of raw tile; only do this once assuming all times are the same
        if not width and not height:
          height, width = get_dims(filename)
          # create tiled grid closest to 960x540
          n_tilesh = math.ceil(height / TARGET_TILE_HEIGHT)
          n_tilesw = math.ceil(width / TARGET_TILE_WIDTH)

        head, tail = os.path.split(filename)
        key = tail.split('.')[0]

        if row['Category'].upper() not in group_map.keys():
          continue

        # if haven't converted tiles to smaller grid, convert
        if key not in frame_dict.keys():
          frame_dict[key] = 1
          image_file = '{0}/{1}_{2:02}.png'.format(png_dir, key, 0)
          # http://www.imagemagick.org/Usage/crop/#crop_equal
          if not os.path.exists(image_file):
            print('Converting {0} into tile'.format(filename))
            os.system('/usr/local/bin/convert "{0}" -crop {1}x{2}@ +repage +adjoin -quality 100%% "{3}/{4}_%02d.png"'.format(filename, n_tilesw, n_tilesh, png_dir, key));
          if not actual_height and not actual_width:
            actual_height, actual_width = get_dims(image_file)

        print('Processing row {0} filename {1} annotation {2}'.format(index, filename, row['Category']))
        a = aesa_annotation(category=row['Category'], centerx=row['CentreX'], centery=row['CentreY'],
                            measurement=row['Measurement'], mtype=row['Type'],
                            index=index)

        a = correct(a)
        a = a._replace(category=group_map[a.category.upper()])
        obj, filename = convert_annotation(filename, png_dir, a, height, width, actual_height, actual_width)
        head, tail = os.path.split(filename)
        key = tail.split('.')[0]
        if key not in annotation_dict.keys():
          a = defaultdict(list)
          a['folder'] = conf.COLLECTION
          a['filename'] = tail
          a['size'] = {}
          a['size']['width'] = actual_width
          a['size']['height'] = actual_height
          a['size']['depth'] = conf.DEPTH
          a['source'] = {}
          a['source']['image'] = 'AESA'
          a['source']['database'] = conf.DATABASE
          annotation_dict[key] = a

        print('Appending object to key {0}'.format(key))
        annotation_dict[key]['object'].append(obj)
        ntest += 1
        if ntest > 100:
          break;

      except Exception as ex:
        failed_file.write("Error cropping annotation row {0} filename {1} \n".format(index, filename))

    ensure_dir(conf.ANNOTATION_DIR)
    for key, data in annotation_dict.items():
      print('Key {0}'.format(key))
      try:
        tf_example, num_objs = dict_to_tf_example(data, conf.DATA_DIR, label_map_dict, png_dir)

        ttl_objs += num_objs
        xml = dicttoxml(data, custom_root='annotation', attr_type=False)

        dom = parseString(xml)
        pretty_xml_as_string = dom.toprettyxml()

        # remove empty lines
        pretty_xml_as_string = os.linesep.join([s for s in pretty_xml_as_string.splitlines() if s.strip()])
        xml_file = os.path.join(conf.ANNOTATION_DIR, '{0}.xml'.format(key))
        with open(xml_file, 'w') as f2:
          f2.write(pretty_xml_as_string)
        f2.close()

        # copy tile to the target directory
        #src = os.path.join(png_dir, data['filename'])
        #dst = os.path.join(png_collection_dir, data['filename'])
        #shutil.copy(src, dst)
        print('{0} objects found in {1}'.format(num_objs, xml_file))

      except Exception as ex:
        print('Exception {0}'.format(ex))
        continue

  except Exception as ex:
    print(ex)

  print('{0} total objects found in {1} frames'.format(ttl_objs, len(annotation_dict.keys())))
  print('Done')
