#!/usr/bin/env python
__author__ = 'Danelle Cline'
__copyright__ = '2017'
__license__ = 'GPL v3'
__contact__ = 'dcline at mbari.org'
__doc__ = '''

Converts AESA annotations into a Tensorflow record for object detection tests 

@var __date__: Date of last svn commit
@undocumented: __doc__ parser
@status: production
@license: GPL
'''
from xml.dom.minidom import parseString
import hashlib
import conf
import io
import os
import sys
import tensorflow as tf
from utils import ensure_dir
from annotation import Annotation

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


if __name__ == '__main__':

  ensure_dir(conf.DATA_DIR)
  ensure_dir(conf.ANNOTATION_DIR)
  ttl_objs = 0

  try:

    label_map_dict = label_map_util.get_label_map_dict(conf.LABEL_PATH_PATH)
    a = Annotation(conf.GROUP_FILE)
    a.generate_tiles()
    a.aggregate()

    for key, data in a.get_dict().items():
      print('Key {0}'.format(key))
      try:
        tf_example, num_objs = dict_to_tf_example(data, conf.DATA_DIR, label_map_dict, conf.PNG_DIR)

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

  print('{0} total objects found in {1} frames'.format(ttl_objs, len(a.get_dict().keys())))
  print('Done')
