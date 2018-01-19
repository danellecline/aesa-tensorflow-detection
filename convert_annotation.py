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
from lxml import etree
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


if __name__ == '__main__':

    ensure_dir(conf.DATA_DIR)
    ensure_dir(conf.ANNOTATION_DIR)
    ensure_dir(conf.PNG_DIR)
    ensure_dir(conf.TILE_PNG_DIR)
    ttl_objs = 0

    try:

        label_map_dict = label_map_util.get_label_map_dict(conf.LABEL_PATH_PATH)
        a = Annotation(conf.GROUP_FILE)
        a.generate_tiles()
        a.aggregate()

        for key, data in a.get_dict().items():
            print('Key {0}'.format(key))
            try:
                xml = dicttoxml(data, custom_root='annotation', attr_type=False)

                # dicttoxml doesn't quite put the xml into the correct format for TF
                # remove tag object, and replace each child tag item with object
                root = etree.fromstring(xml)
                etree.strip_tags(root, 'object')
                for element in root.getiterator():
                  if 'item' in element.tag :
                    element.tag = 'object'

                dom = parseString(etree.tostring(root))
                pretty_xml_as_string = dom.toprettyxml()
                # remove empty lines
                pretty_xml_as_string = os.linesep.join([s for s in pretty_xml_as_string.splitlines() if s.strip()])
                xml_file = os.path.join(conf.ANNOTATION_DIR, '{0}.xml'.format(key))
                with open(xml_file, 'w') as f2:
                    f2.write(pretty_xml_as_string)
                f2.close()

            except Exception as ex:
              print('Exception {0}'.format(ex))
              continue

    except Exception as ex:
        print(ex)

    print('Done')
