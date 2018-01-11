#!/usr/bin/env python
__author__ = 'Danelle Cline'
__copyright__ = '2017'
__license__ = 'GPL v3'
__contact__ = 'dcline at mbari.org'
__doc__ = '''

Reads in AESA annotation file and aggregates the annotations by frame. 
Converts the tiles into a grid and adjusts the annotation to map to this new coordinate system.

@var __date__: Date of last svn commit
@undocumented: __doc__ parser
@status: production
@license: GPL
'''
import cv2
import numpy
import pandas as pd
import numpy as np
import os
import math
import conf
import image_utils
from collections import defaultdict
from collections import namedtuple
from shutil import copyfile

class Annotation():
  frame_dict = {}
  annotation_dict = {}
  group_map = {}
  tile_width = None
  tile_height = None
  image_width = None
  image_height = None
  scale_x = 1.0
  scale_y = 1.0
  annotations = []

  aesa_annotation = namedtuple("Annotation", ["category", "group", "x", "y", "pad", "filename", "length"])

  def __init__(self, group_file = None):
      print('Parsing ' + conf.ANNOTATION_FILE)
      self.df_annotations = pd.read_csv(conf.ANNOTATION_FILE, sep=',')
      self.csv_file = conf.ANNOTATION_FILE[conf.ANNOTATION_FILE.rfind("/") + 1:]
      self.frame_dict = {}

      if group_file:
          print('Parsing ' + group_file)
          self.df_group = pd.read_csv(group_file, sep=',')
          for index, row in self.df_group.iterrows():
              if np.isnan(float(row['coarse_class'])):
                  print('Category {0} does not have a group'.format(row['category']))
              else:
                  print('Grouping category {0} as GROUP{1}'.format(row['category'], int(row['coarse_class'])))
                  self.group_map[row['category']] = 'GROUP{:d}'.format(int(row['coarse_class']))

      ntest = 0
      for index, row in self.df_annotations.iterrows():
          try:
              f = row['FileName']
              if conf.FILE_FORMAT:
                filename = os.path.join(conf.TILE_DIR, '{0}{1}.jpg'.format(conf.FILE_FORMAT, f))
              else:
                filename = os.path.join(conf.TILE_DIR, f)
              print('Processing row {0} filename {1} annotation {2}'.format(index, filename, row['Category']))

              category = row["Category"].upper()
              group = None
              try:
                group = row["group"].upper()
              except:
                pass
              x = int(row["CentreX"])
              y = int(row["CentreY"])
              length = float(row["Measurement"])
              if np.isnan(length):
                  length = conf.MIN_CROP
              pad = (length / 100.0) * conf.PAD_PERCENTAGE
              a = self.aesa_annotation(category=category, group=group, x=x, y=y, pad=pad, filename=filename, length=length)

              if group_file and a.category.upper() not in self.group_map.keys():
                continue

              a = self.correct(a)
              if group_file:
                a = a._replace(category=self.group_map[a.category.upper()])
              self.annotations.append(a)
              ntest += 1
              if ntest > 10:
                break;

          except Exception as ex:
              print("Error processing annotation row {0} filename {1} \n".format(index, filename))

  def aggregate(self):
      '''
        Aggregates each annotation object into a dictionary keyed by frame.
        Requires images to be generated first with generate_tiles().
      :return:
      '''
      for a in self.annotations:
          try:
              obj, filename = self.annotation_to_dict(a)
              head, tail = os.path.split(filename)
              key = tail.split('.')[0]
              if key not in self.annotation_dict.keys():
                  a = defaultdict(list)
                  a['folder'] = conf.COLLECTION
                  a['filename'] = tail
                  a['size'] = {}
                  a['size']['width'] = conf.TARGET_TILE_WIDTH
                  a['size']['height'] = conf.TARGET_TILE_HEIGHT
                  a['size']['depth'] = conf.DEPTH
                  a['source'] = {}
                  a['source']['image'] = 'AESA'
                  a['source']['database'] = conf.DATABASE
                  self.annotation_dict[key] = a

              print('Appending object to key {0}'.format(key))
              self.annotation_dict[key]['object'].append(obj)
          except Exception as ex:
              print("Error converting annotation {0} to dictionary \n".format(a.filename))
              continue

  def get_dict(self):
      '''
       Helper function to return annotation dictionary
      :return: annotation dictionary; a dictionary with objects keyed by frame
      '''
      return self.annotation_dict

  def generate_tiles(self):
      '''
       Generates tiles from each image. This converts the raw 10x1 tiled images using a  grid closest to
       conf.TARGET_TILE_WIDTH by conf.TARGET_TILE_HEIGHT
       This is required because the raw 10x1 tiled images are very large, and will overrun the memory during
       training the CNN
      :param self:
      :return:
      '''
      for a in self.annotations:
          try:

              # filename = '{0}/data/{1}'.format(os.getcwd(), 'M56_10441297_12987348614060.jpg' )

              # get image height and width of raw tile; only do this once assuming all times are the same
              if self.image_width is None and self.image_height is None:
                  self.image_height, self.image_width = image_utils.get_dims(a.filename)
                  n_tilesh = 10
                  n_tilesw = 1

              head, tail = os.path.split(a.filename)
              key = tail.split('.')[0]

              # if haven't converted tiles to smaller grid, convert
              if key not in self.frame_dict.keys():
                  self.frame_dict[key] = 1
                  image_file = '{0}/{1}_{2:02}.png'.format(conf.TILE_PNG_DIR, key, 0)
                  # http://www.imagemagick.org/Usage/crop/#crop_equal
                  if not os.path.exists(image_file):
                      print('Converting {0} into tiles'.format(a.filename))
                      os.system('/usr/local/bin/convert "{0}" -crop {1}x{2}@ +repage +adjoin -quality 100%% "{3}/{4}_%02d.png"'
                              .format(a.filename, n_tilesw, n_tilesh, conf.TILE_PNG_DIR, key));
                  if not self.tile_height and not self.tile_width:
                      self.tile_height, self.tile_width = image_utils.get_dims(image_file)
                  self.bins_y = numpy.arange(start=0, stop=self.image_height - self.tile_height, step=self.tile_height - 1)
                  self.scale_x = conf.TARGET_TILE_WIDTH/self.tile_width
                  self.scale_y = conf.TARGET_TILE_HEIGHT/self.tile_height

          except Exception as ex:
            print("Error converting image {0} \n".format(a.filename))


  def correct(self, annotation):
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


  def annotation_to_dict(self, annotation):
    '''
     Corrects annotation bounding box and converts annotation to dictionary
    :param raw_file:  path to file with image
    :param annotation: annotation tuple
    :return: the annotation dictionary and the image filename the annotation is associated with
    '''

    head, tail = os.path.split(annotation.filename)
    stem = tail.split('.')[0]
    posx = 1
    posy = numpy.digitize(annotation.y, self.bins_y)

    # calculate what image index the annotations should fall into
    index = (posy - 1) * int(self.image_width / self.tile_width) + (posx - 1)

    # http://www.imagemagick.org/Usage/crop/#crop_equal
    #print('Index: {0} center x {1} centery {2}'.format(index, values_x, values_y))
    src_file = '{0}/{1}_{2:02}.png'.format(conf.TILE_PNG_DIR, stem, index)
    dst_file = '{0}/{1}_{2:02}.png'.format(conf.PNG_DIR, stem, index)
    temp_file = '/tmp/temp.png'

    copyfile(src_file, temp_file)

    length = conf.MIN_CROP
    if annotation.length < conf.MIN_CROP:
      length = conf.MIN_CROP  # to prevent generating really small crops
    center_x = annotation.x - (posx - 1) * self.tile_width
    center_y = annotation.y - (posy - 1) * self.tile_height
    tlx = int(max(0, int(center_x - (length/2) - annotation.pad)))
    tly = int(max(0, int(center_y - (length/2) - annotation.pad)))
    brx = int(min(self.tile_width ,center_x + (length/2) + annotation.pad))
    bry = int(min(self.tile_height ,center_y + (length/2) + annotation.pad))
    img = cv2.imread(temp_file)
    crop_img = img[tly:bry, tlx:brx]
    gray_image = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    cv2.rectangle(img, (tlx, tly), (brx, bry), (0, 255, 0), 3)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, annotation.category, (tlx, tly), font, 2, (255, 255, 255), 2)
    cv2.putText(img, str(index), (brx, bry), font, 2, (255, 255, 255), 2)
  
    ret2, th = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    erode = cv2.erode(th, kernel, iterations=1)
    clean = cv2.dilate(erode, kernel2, iterations=1) 
    
    # first try Otsu
    cv2.imshow('Otsu', clean)
    found, x, y, w, h = image_utils.find_object(clean, crop_img)
  
    if not found and 'CNIDARIA' not in annotation.category and 'OPHIUR' not in annotation.category:
        # Next try threshold in increments of 2
        for thresh in range(3, 13, 2):
          th = cv2.adaptiveThreshold(th, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                     cv2.THRESH_BINARY, thresh, 2)
          #cv2.imshow('Thresh {0}'.format(thresh), th)
          found, x, y, w, h = image_utils.find_object(th, crop_img)
          if found:
            break;
  
    if found:
      # adjust the actual bounding box with new localized bounding box coordinates
      tlx += x;
      tly += y;
      brx = tlx + w;
      bry = tly + h;
  
    cv2.destroyAllWindows()
    cv2.rectangle(img, (tlx, tly), (brx, bry), (0, 255, 0), 3)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, annotation.category, (tlx, tly), font, 2, (255, 255, 255), 2)
    cv2.namedWindow('Annotation', cv2.WINDOW_NORMAL)
    cv2.imshow("Annotation", img)
    cv2.resizeWindow('Annotation', 950, 540)
    cv2.waitKey(1000)
    obj = {}
    obj['name'] = annotation.category
    obj['difficult'] = conf.DIFFICULT
    obj['truncated'] = 0
    obj['pose'] = conf.POSE
    obj['bndbox'] = {}
    obj['bndbox']['xmin'] = int(self.scale_x*tlx)
    obj['bndbox']['ymin'] = int(self.scale_y*tly)
    obj['bndbox']['xmax'] = int(self.scale_x*brx)
    obj['bndbox']['ymax'] = int(self.scale_y*bry)

    if not os.path.exists(dst_file):
      resized_img = cv2.resize(img, (conf.TARGET_TILE_WIDTH, conf.TARGET_TILE_HEIGHT))
      cv2.imwrite(dst_file, resized_img)

    return obj, dst_file
