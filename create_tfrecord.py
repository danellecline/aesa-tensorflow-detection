#!/usr/bin/env python
import re

__author__    = 'Danelle Cline'
__copyright__ = '2017'
__license__   = 'GPL v3'
__contact__   = 'dcline at mbari.org'
__doc__ = '''

Reads in AESA annotation xml files amd  converts into a Tensorflow record for object detection tests

@var __date__: Date of last svn commit
@undocumented: __doc__ parser
@status: production
@license: GPL
'''

import hashlib
import conf
import io
import os
import logging
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'tensorflow_models', 'research'))
from lxml import etree
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

SETS = ['train', 'val', 'trainval', 'test']


def process_command_line():
  '''
  Process command line
  :return: args object
  '''

  import argparse
  from argparse import RawTextHelpFormatter

  examples = 'Examples:' + '\n\n'
  examples += 'Create record for xml files :\n'
  examples += '{0} --data_dir /Users/dcline/Dropbox/GitHub/mbari-tensorflow-detection/data/ --collection ' \
              'M56_1000x1000_by_group M535455_1000x1000_by_group --output_path M5354556_1000x1000_by_group_test.record \
               --label_map_path aesa_group_label_map.pbtxt -set test '.format(sys.argv[0])
  parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
                                   description='Creates Tensorflow Record object for annotated data',
                                   epilog=examples)
  parser.add_argument('-d', '--data_dir', action='store', help='Root directory to raw dataset', required=True)
  parser.add_argument('-c', '--collection', action='store', help='List of space seprated collections to combine.'
                                                                 ' Also the subdirectory name for the raw dataset',
                      nargs='*', default=['M535455_1000x1000_by_group', 'M56_1000x1000_by_group'], required=True)
  parser.add_argument('-o', '--output_path', action='store', help='Path to output TFRecord', required=True)
  parser.add_argument('-l', '--label_map_path', action='store', help='Path to label map proto', required=True)
  parser.add_argument('-s', '--set', action='store', help='Convert training set, validation set or merged set.',
                      required=True)
  parser.add_argument('--labels', action='store',
                      help='List of space separated labels to load. Must be in the label map proto', nargs='*',
                      required=False)

  args = parser.parse_args()
  return args


def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       labels,
                       image_subdirectory='imgs'):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding dataset
    label_map_dict: A map from string label names to integers ids.
    labels: list of labels to include in the record
    image_subdirectory: String specifying subdirectory within the
      PASCAL dataset directory holding the actual image data.

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
  label_count = {}
  for item, k in label_map_dict.items():
    label_count[item] = 0

  for obj in data['object']:
    if labels and obj['name'] not in labels:
      continue
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

    label_count[obj['name']] += 1

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
  return example, label_count


def main(_):
  args = process_command_line()

  if args.set not in SETS:
    raise ValueError('set must be in : {}'.format(SETS))

  output = os.path.join(args.data_dir, args.output_path)

  # touch the file if it doesn't already exist
  if not os.path.exists(output):
    with open(output, 'a'):
      os.utime(output)

  writer = tf.python_io.TFRecordWriter(output)
  label_map_dict = label_map_util.get_label_map_dict(os.path.join(args.data_dir, args.label_map_path))
  label_count = {}
  for item, key in label_map_dict.items():
    label_count[item] = 0
  for c in args.collection:
    print('Reading from {0} dataset.'.format(c))
    examples_path = os.path.join(args.data_dir, c, args.set + '.txt')
    png_dir = '{0}/{1}/PNGImages'.format(args.data_dir, c)
    annotations_dir = '{0}/{1}/Annotations'.format(args.data_dir, c)

    with open(examples_path) as fid:
      lines = fid.readlines()
      examples_list = [line.strip() for line in lines]

    for idx, example in enumerate(examples_list):
      if idx % 10 == 0:
        logging.info('Processing image %d of %d', idx, len(examples_list))
      file = os.path.join(annotations_dir, example)
      with open(file, 'r') as fid:
        xml_str = fid.read()
      try:
        xml = etree.fromstring(xml_str)
      except Exception as ex:
        print(ex)
      data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
      tf_example, count = dict_to_tf_example(data, args.data_dir, label_map_dict, args.labels, png_dir)
      if tf_example:
        for key, item in count.items():
          label_count[key] += item
        writer.write(tf_example.SerializeToString())
      else:
        logging.warn('No objects found in {0}'.format(example))

  writer.close()
  ttl_objs = 0
  for key, item in label_count.items():
    print('{0} {1}'.format(key, item))
    ttl_objs += item
  print('Done. Found {0} examples in {1} set'.format(ttl_objs, args.set))


if __name__ == '__main__':
  tf.app.run()
