#!/usr/bin/env python
import re

__author__    = 'Danelle Cline'
__copyright__ = '2017'
__license__   = 'GPL v3'
__contact__   = 'dcline at mbari.org'
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
from bs4 import BeautifulSoup
from collections import defaultdict         
from collections import namedtuple
import argparse
import subprocess 
import cv2 
import hashlib 
import io
import os 
import sys
import tensorflow as tf
sys.path.append(os.path.join(os.path.dirname(__file__), 'tensorflow_models', 'research'))

from PIL import Image
from dicttoxml import dicttoxml
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

def process_command_line():
    from argparse import RawTextHelpFormatter

    examples = 'Examples:' + '\n\n'
    examples += sys.argv[0] + """--tile_dir /Volumes/ScratchDrive/AESA/M56 tiles/raw/ --collection M56
              --labels /Users/dcline/Dropbox/GitHub/aesa-tensorflow-detection/data/aesa_k5_label_map.pbtx \n
              --out_path /Users/dcline/Dropbox/GitHub/aesa-tensorflow-detection/data/ \n
              --data_dir /Users/dcline/Dropbox/GitHub/aesa-tensorflow-detection/data/ \n
              --group_file /Volumes/ScratchDrive/AESA/hierarchy_group_k5.csv \n
              --annotation_file /Volumes/ScratchDrive/AESA/M56_Annotations_v13.csv \n"""
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
                                     description='Extract cropped images from tiles and associated annotations',
                                     epilog=examples)
    parser.add_argument('-d', '--data_dir',action='store', help='Root directory to raw dataset', default=os.path.join(os.getcwd(),'data'), required=False)
    parser.add_argument('-c', '--collection', action='store', help='Name of the collection. Also the subdirectory name '
                                                                   'for the generated images', default='M56',  required=False)
    parser.add_argument('-o', '--out_path', action='store', help='Path to output TFRecord', default=os.path.join(os.getcwd(), 'data', 'M56_test.record'), required=False)
    parser.add_argument('-l', '--label_map_path', action='store', help='Path to label map proto', default=os.path.join(os.getcwd(), 'aesa_k5_label_map.pbtxt', ), required=False)
    parser.add_argument('--labels', action='store',
                        help='List of space separated labels to load. Must be in the label map proto', nargs='*',
                        required=False)
    parser.add_argument('--tile_dir', type=str, required=False, default='/Volumes/ScratchDrive/AESA/M56 tiles/raw/', help="Path to folders of raw tiled images.")

    parser.add_argument('--annotation_file', type=str, required=False, default=os.path.join(os.getcwd(), 'data/annotations/M56_Annotations_v13.csv'), help="Path to annotation file.")
    parser.add_argument('--group_file', type=str, required=False, default=os.path.join(os.getcwd(), 'data/annotations/hierarchy_group_k5.csv'), help="Path to annotation group file.")
    parser.add_argument('--file_format', default='M56_10441297_', type=str, required=False, help="Alternative file prefix to use for calculating the associated frame the annotation is from, e.g. M56_10441297_%d.jpg'")
    parser.add_argument('--strip_trailing_dot', action='store_true', required=False, help="Strip trailing .'")
    args = parser.parse_args()
    return args

def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       labels,
                       image_subdirectory='imgs'):
  """Convert csv derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding fields for a single image
    dataset_directory: Path to root directory holding dataset
    label_map_dict: A map from string label names to integers ids. 
    labels: list of labels to include in the record
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
    if labels and obj['name'] not in labels:
        continue
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
  
def find_object(image_bin, image_color):
 
    invert = cv2.bitwise_not(image_bin);  
    cv2.imshow('invert', invert)
               
    # get blobs
    im, contours, heirachy = cv2.findContours(invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.drawContours(invert, contours, -1, (0, 255, 0), 7)
    cv2.imshow('contours', img)
    
    # keep valid contours 
    valid_contours = []
    cnt = 0
    for c in contours:
        try:
            # get rectangle bounding contour
            [x,y,w,h] = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            print("area {0} x {1} y {2} w {3} h {4}".format(area, x, y, w, h))
            # get valid areas, not blobs along the edge or noise
            if area > 1000 and area < 6000 :
              pt = [float(y), float(y)]
              cn = contours[cnt]
              img = cv2.drawContours(image_color, [cn], 0, (255, 0, 0), 5)
              cv2.imshow('Possible', img)
              valid_contours.append(c) 
            cnt += 1

        except Exception as ex:
            print(ex)
            
    if len(valid_contours) > 0:
        # find largest contour in mask
        c = max(valid_contours, key=cv2.contourArea)  
        [x,y,w,h] = cv2.boundingRect(c)
        return True, x, y, w, h
        
    return False, -1, -1 , -1, -1


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
        crop_pixels = 2*int(float(annotation.measurement))
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

    index = (posy - 1)* int(image_width/tile_width) + (posx - 1)

    print('Index: {0} center x {1} centery {2}'.format(index, values_x, values_y))

    image_file = '{0}_{1:02}.png'.format(stem, index)
 
    img = cv2.imread('{0}/{1}'.format(img_dir, image_file))
    center_x = values_x - (posx - 1) * tile_width
    center_y = values_y - (posy - 1) * tile_height
    tlx = max(0, int(center_x - crop_pixels/2))
    tly = max(0, int(center_y - crop_pixels/2)) 
    if a.mtype == 'Length':
        brx = min(tile_width, int(tlx + crop_pixels))
        bry = min(tile_height, int(tly + crop_pixels))
    else:
        brx = min(tile_width, int(tlx + crop_pixels/2))
        bry = min(tile_height, int(tly + crop_pixels/2))

    #display_annotation(annotation, brx, bry, img, tlx, tly)
    obj = {}
    obj['name'] = a.category
    obj['difficult'] = 0
    obj['truncated'] = 0
    obj['pose'] = 'Unspecified'
    obj['bndbox'] = {}
    obj['bndbox']['xmin'] = tlx
    obj['bndbox']['ymin'] = tly
    obj['bndbox']['xmax'] = brx
    obj['bndbox']['ymax'] = bry
    print('done')
    return obj, image_file
 
def display_annotation(annotation, brx, bry, img, tlx, tly):
    cv2.rectangle(img, (tlx, tly), (brx, bry), (0, 255, 0), 3)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, annotation.category, (tlx, tly), font, 2, (255, 255, 255), 2)
    cv2.imshow("Annotation", img) 

def get_dims(image):
  """
  get the height and width of a tile
  :param image: the image file
  :return: height, width
  """
  cmd = '/usr/local/bin/identify "{0}"'.format(image)
  subproc = subprocess.Popen(cmd, env=os.environ, shell=True, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                             stdout=subprocess.PIPE)
  out, err = subproc.communicate()
  # get image height and width of raw tile
  p = re.compile(r'(?P<width>\d+)x(?P<height>\d+)')
  match = re.search(pattern=p, string=str(out))
  if (match):
      width = int(match.group("width"))
      height = int(match.group("height"))
      return height, width

  raise Exception('Cannot find height/width for image {0}'.format(image))

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

def ensure_dir(d):
  """
  ensures a directory exists; creates it if it does not
  :param fname:
  :return:
  """
  if not os.path.exists(d):
    os.makedirs(d)

if __name__ == '__main__':
  args = process_command_line()

  ensure_dir(args.data_dir)
  failed_file = open(os.path.join(args.data_dir, 'failed_crops.txt'), 'w')
  writer = tf.python_io.TFRecordWriter(args.out_path)
  png_dir = os.path.join(args.data_dir, 'imgs')
  ensure_dir(png_dir)
  label_map_dict = label_map_util.get_label_map_dict(args.label_map_path)
  aesa_annotation = namedtuple("Annotation", ["category", "centerx", "centery", "mtype", "measurement", "index"])

  try:
    print('Parsing ' + args.annotation_file)
    df = pd.read_csv(args.annotation_file, sep=',')
    print('Parsing ' + args.group_file)
    df_group = pd.read_csv(args.group_file, sep=',')
    group_map = {}
    for index,row in df_group.iterrows():
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

    for index, row in df.iterrows():
      try:
        #if index > 1:
        #  break;
        f = row['FileName'] 
        if args.file_format:
          filename = os.path.join(args.tile_dir, '{0}{1}.jpg'.format(args.file_format, f))
        else:
          filename = os.path.join(args.tile_dir, f)

        #filename = '{0}/data/{1}'.format(os.getcwd(), 'M56_10441297_12987348614060.jpg' )
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
            # http://www.imagemagick.org/Usage/crop/#crop_equal
            if not os.path.exists('{0}/{1}_86.png'.format(png_dir, key)):
                print("Converting {0} to tiles".format(filename))
                os.system('/usr/local/bin/convert "{0}" -crop {1}x{2}@ +repage +adjoin -quality 100%% "{3}/{4}_%02d.png"'.
                  format(filename, n_tilesw, n_tilesh, png_dir, key));
            image_file = '{0}/{1}_{2:02}.png'.format(png_dir, key, 0)

            if not actual_height and not actual_width:
                actual_height, actual_width = get_dims(image_file)

        print('Processing row {0} filename {1} annotation {2}'.format(index, filename, row['Category']))
        a = aesa_annotation(category=row['Category'],centerx=row['CentreX'], centery=row['CentreY'],
                            measurement=row['Measurement'], mtype=row['Type'],
                            index=index)

        a = correct(a)
        a = a._replace(category=group_map[a.category.upper()])
        obj, filename = convert_annotation(filename, png_dir, a, height, width, actual_height, actual_width)
        head, tail = os.path.split(filename)
        key = tail.split('.')[0]
        if key not in annotation_dict.keys():
            a = defaultdict(list)
            a['folder'] = 'M56'
            a['filename'] = tail
            a['size'] = {}
            a['size']['width'] = actual_width
            a['size']['height'] = actual_height
            a['size']['depth'] = 3
            a['source'] = {}
            a['source']['image'] = 'AESA'
            a['source']['database'] = 'Unknown'
            annotation_dict[key] = a

        annotation_dict[key]['object'].append(obj)
        
      except Exception as ex:
          failed_file.write("Error cropping annotation row {0} filename {1} \n".format(index, filename))

    ttl_objs = 0
    png_collection_dir = os.path.join(args.data_dir, args.collection, 'PNGImages')
    ensure_dir(png_collection_dir)

    for key, data in annotation_dict.items():
        tf_example, num_objs = dict_to_tf_example(data, args.data_dir, label_map_dict, args.labels, png_dir)

        if tf_example:
            ttl_objs += num_objs
            writer.write(tf_example.SerializeToString())
            xml = dicttoxml(data, custom_root='annotation', attr_type=False)
            xml_out = os.path.join(args.data_dir, args.collection, 'Annotations')
            dom = parseString(xml)
            pretty_xml_as_string = dom.toprettyxml()
            # remove empty lines
            pretty_xml_as_string = os.linesep.join([s for s in pretty_xml_as_string.splitlines() if s.strip()])
            ensure_dir(xml_out)
            with open(os.path.join(xml_out,'{0}.xml'.format(key)), 'w') as f2:
                f2.write(pretty_xml_as_string)
            f2.close()
            # copy tile to the target directory
            src = os.path.join(png_dir, data['filename'] )
            dst = os.path.join(png_collection_dir, data['filename'])
            shutil.copy(src, dst)
        else:
            print('No objects found in {0}'.format(tf_example))

  except Exception as ex:
      print(ex)

  print('Done')
