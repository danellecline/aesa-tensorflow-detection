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
  annotations = []

  aesa_annotation = namedtuple("Annotation", ["category", "group", "heirarchical_group", "x", "y", "pad", "filename", "length"])
  image_metadata = namedtuple("Metadata", ["tile_width", "tile_height", "bins_x", "bins_y", "image_width", "image_height", "crop_left", "crop_right"])

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
                  # GROUP1, GROUP2, 1-based indexing
                  print('Grouping category {0} as GROUP{1}'.format(row['category'], int(row['coarse_class']) + 1))
                  self.group_map[row['category']] = 'GROUP{:d}'.format(int(row['coarse_class']) + 1)

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
              a = self.aesa_annotation(category=category, group=group, heirarchical_group=None, x=x, y=y, pad=pad, filename=filename, length=length)

              if group_file and a.category.upper() not in self.group_map.keys():
                continue

              a = self.correct(a)
              if group_file:
                a = a._replace(heirarchical_group=self.group_map[a.category.upper()])

              if a.category in conf.OPTIMIZED_CATEGORIES:
                self.annotations.append(a)
              ntest += 1
              if ntest > 1000:
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
              print("Error converting annotation {0} to dictionary. {1}".format(a.filename, ex))
              continue

  def get_dict(self):
      '''
       Helper function to return annotation dictionary
      :return: annotation dictionary; a dictionary with objects keyed by frame
      '''
      return self.annotation_dict

  def generate_tiles(self):
      '''
       Generates tiles from each image. This converts the raw 10x1 or 1x10 tiled images to a grid.
       This is required because the raw 10x1 tiled images are very large, and will overrun the memory during
       training the CNN
      :param self:
      :return:
      '''
      for a in self.annotations:
          try:

              n_tilesh = 10
              n_tilesw = 1

              head, tail = os.path.split(a.filename)
              key = tail.split('.')[0]

              # if haven't converted tiles to smaller grid, convert
              if key not in self.frame_dict.keys():
                  image_height, image_width = image_utils.get_dims(a.filename)
                  if image_height < 300 or image_width < 300:
                    raise Exception('image {0} height {1} or width {2} too small'.format(a.filename, image_width, image_height))

                  image_file = '{0}/{1}_{2:02}.png'.format(conf.TILE_PNG_DIR, key, 0)
                  # http://www.imagemagick.org/Usage/crop/#crop_equal
                  if not os.path.exists(image_file) and image_width < image_height:
                      print('Converting {0} into tiles'.format(a.filename))
                      if image_height > image_width:
                          os.system('/usr/local/bin/convert "{0}" -crop {1}x{2}@ +repage +adjoin -quality 100%% "{3}/{4}_%02d.png"'
                              .format(a.filename, n_tilesw, n_tilesh, conf.TILE_PNG_DIR, key));
                      else:
                          os.system('/usr/local/bin/convert "{0}" -crop {2}x{1}@ +repage +adjoin -quality 100%% "{3}/{4}_%02d.png"'
                              .format(a.filename, n_tilesw, n_tilesh, conf.TILE_PNG_DIR, key));

                  tile_height, tile_width = image_utils.get_dims(image_file)
                  if image_height > image_width:
                      bins_x = numpy.arange(start=0, stop=image_width - tile_width, step=tile_width - 1)
                      bins_y = numpy.arange(start=0, stop=image_height - tile_height, step=tile_height - 1)
                  else:
                      bins_x = numpy.arange(start=0, stop=image_width - tile_width, step=tile_width - 1)
                      bins_y = numpy.arange(start=0, stop=image_height - tile_height, step=tile_height - 1)

                  self.frame_dict[key] = self.image_metadata(tile_width=tile_width, tile_height=tile_height,
                                                         image_width=image_width, image_height=image_height,
                                                         bins_x=bins_x, bins_y=bins_y, crop_left=0, crop_right=0)

          except Exception as ex:
            print("Error converting image {0}. Exception {1}".format(a.filename, ex))
            self.frame_dict[key] = self.image_metadata(tile_width=-1, tile_height=-1, image_width=-1, image_height=-1, bins_x=None, bins_y=None, crop_right=0, crop_left=0)


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

      if 'Polychaete1' in annotation.category:
          annotation.category = 'Polycheata1'
          # annotation.group = 'Polychaeta'

      return annotation

  def crop(self, binary_img, tile_file, tile_width, tile_height, dst_file):
    '''
     Crop the tile image to remove black mosaic artifacts
    :param binary_img: binary image to extract tile edge artifacts from
    :param tile_file: tile to crop
    :param dst_file: destimation file for crop
    :return: amount cropped from left and right side
    '''
    from scipy.cluster.vq import kmeans, vq
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

    # get  blobs
    dilate = cv2.dilate(binary_img, kernel, iterations=3)
    cv2.imwrite('dilate.jpg', dilate)
    img, contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # keep valid contours
    x_l = []
    y_l = []
    w_l = []
    x_c = []
    for contour in contours:
      # get rectangle bounding contour
      [x, y, w, h] = cv2.boundingRect(contour)
      area = w * h
      #print("area {0} x {1} y {2} w {3} h {4}".format(area, x, y, w, h))
      # get valid areas
      if area > 10000 and x == 0 or y + h == tile_height:
        w_l.append(w)
        x_l.append(x)
        y_l.append(y)
        pt = [float(x), float(x)]
        x_c.append(pt)

    crop_left = 0
    crop_right = 0

    # should be two groups: one on left and one on right
    data = np.array(x_c)
    centroids, _ = kmeans(np.array(x_c), 2, iter=20)
    idx, _ = vq(data, centroids)
    centroids_x = centroids[:,0]

    if len(centroids_x) == 2 :
      group = 0
      distance = np.diff(sorted(centroids_x))
      if int(distance) > 1500:
        # for each group, find extents for the boxes
        for c in centroids:
          idx_group = [i for i, j in enumerate(idx) if j == group]
          x_ll = [x_l[i] for i in idx_group]
          w_ll = [w_l[i] for i in idx_group]
          # find the extents in x
          x_1 = min(x_ll)
          width = max(w_ll)

          if x_1 == 0:
            crop_left = width
            tlx = crop_left
          else:
            crop_right = width
          group += 1

      tly = 0
      w = tile_width - crop_left - crop_right
      h = tile_height
      cmd = '/usr/local/bin/convert "{0}" -crop {1}x{2}+{3}+{4} "{5}"'.format(tile_file, w, h, tlx, tly, dst_file);
      print('Running {0}'.format(cmd))
      os.system('/usr/local/bin/convert "{0}" -crop {1}x{2}+{3}+{4} "{5}"'.format(tile_file, w, h, tlx, tly, dst_file));

    return crop_left, crop_right

  def annotation_to_dict(self, annotation):
    '''
     Corrects annotation bounding box and converts annotation to dictionary
    :param annotation: annotation tuple
    :return: the annotation dictionary and the image filename the annotation is associated with
    '''

    head, tail = os.path.split(annotation.filename)
    key = tail.split('.')[0]
    metadata = self.frame_dict[key]
    if metadata.bins_x is None:
      raise Exception('{0} missing valid metadata'.format(key))

    # calculate what tile index the annotations should fall into
    bin_x = 0
    bin_y = 0
    index = 0
    if len(metadata.bins_x) > 0:
      bin_x = int(numpy.digitize(annotation.x, metadata.bins_x)) - 1
      index = bin_x
    if len(metadata.bins_y) > 0:
      bin_y = int(numpy.digitize(annotation.y, metadata.bins_y)) - 1
      index = bin_y

    if index == 0:
      raise Exception("Skipping zero index Annotation {0}".format(annotation.filename))

    src_file = '{0}/{1}_{2:02}.png'.format(conf.TILE_PNG_DIR, key, index)
    dst_file = '{0}/{1}_{2:02}.png'.format(conf.PNG_DIR, key, index)
    temp_file = '/tmp/temp.png'

    #xml_file = '{0}/{1}.xml'.format(conf.ANNOTATION_DIR, key)
    #if os.path.exists(xml_file):
    #  raise Exception('{0} already created, skipping for performance reasons'.format(xml_file))

    copyfile(src_file, temp_file)
    img = cv2.imread(temp_file)

    # convert coordinates using default tile width/height
    brx, bry, tlx, tly = self.convert_coords(annotation, bin_x, bin_y, metadata.tile_height,
                                             metadata.tile_width, annotation.pad)

    kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    if os.path.exists(dst_file) and metadata.crop_left > 0 or metadata.crop_right > 0:
      metadata_new = metadata
    else :
      yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
      ret, binary_img = cv2.threshold(yuv[:,:,1], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
      crop_left, crop_right = self.crop(binary_img, src_file, metadata.tile_width, metadata.tile_height, dst_file)
      metadata_new = metadata._replace(crop_left=crop_left, crop_right=crop_right)
      self.frame_dict[key] = metadata_new

    # throw annotations out if they fall outside the crop
    if tlx > metadata_new.tile_width - metadata_new.crop_right or tlx < metadata.crop_left:
      raise Exception('{0} falls outside of cropped image'.format(key))

    if conf.OPTIMIZE_BOX == True and (brx - tlx) > conf.MIN_CROP and (bry - tly) > conf.MIN_CROP and \
            annotation.category in conf.OPTIMIZED_CATEGORIES or annotation.group in conf.OPTIMIZED_GROUPS:
      found = False
      brx, bry, tlx, tly = self.convert_coords(annotation, bin_x, bin_y, metadata.tile_height,
                                               metadata.tile_width, annotation.pad)
      crop_img = img[tly:bry, tlx:brx]
      gray_image = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

      cv2.destroyAllWindows()
      ret2, th = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
      clean = cv2.erode(th, kernel3, iterations=3)
      #cv2.imshow('Otsu', clean)
      #cv2.imshow('Cropped', crop_img)
      #cv2.waitKey(2000)
      found, x, y, w, h = image_utils.find_object(clean, crop_img, annotation.length*annotation.length)

      if found:
        # adjust the actual bounding box with new localized bounding box coordinates
        tlx += x;
        tly += y;
        brx = tlx + w;
        bry = tly + h;
        cv2.rectangle(img, (tlx, tly), (brx, bry), (255, 0, 0), 3)

    # adjust coordinates to cropped tile
    tlx -= metadata_new.crop_left
    brx -= metadata_new.crop_left

    '''img = cv2.imread(dst_file)
    cv2.rectangle(img, (tlx, tly), (brx, bry), (0, 255, 0), 3)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, annotation.category, (tlx, tly), font, 2, (255, 255, 255), 2)
    cv2.namedWindow('Annotation', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Annotation', conf.TARGET_TILE_WIDTH, conf.TARGET_TILE_HEIGHT)
    cv2.imshow("Annotation", img)
    cv2.waitKey(3000)'''

    obj = {}
    scale_x = conf.TARGET_TILE_WIDTH/metadata_new.tile_width
    scale_y = conf.TARGET_TILE_HEIGHT/metadata_new.tile_height
    if annotation.heirarchical_group:
      obj['name'] = annotation.heirarchical_group
    else:
      obj['name'] = annotation.category
    obj['difficult'] = conf.DIFFICULT
    obj['truncated'] = 0
    obj['pose'] = conf.POSE
    obj['bndbox'] = {}
    obj['bndbox']['xmin'] = min(int(scale_x*tlx), conf.TARGET_TILE_WIDTH-1)
    obj['bndbox']['ymin'] = min(int(scale_y*tly), conf.TARGET_TILE_HEIGHT-1)
    obj['bndbox']['xmax'] = min(int(scale_x*brx), conf.TARGET_TILE_WIDTH-1)
    obj['bndbox']['ymax'] = min(int(scale_y*bry), conf.TARGET_TILE_HEIGHT-1)

    if obj['bndbox']['xmin'] == obj['bndbox']['xmax'] or obj['bndbox']['ymin'] == obj['bndbox']['ymax']:
      raise Exception("Object too small. Annotation {0}".format(annotation.filename))

    return obj, dst_file

  def convert_coords(self, annotation, posx, posy, t_height, t_width, pad):
    '''
    Converts coordinates from annotation to tile coordinates
    :param annotation:
    :param posx:
    :param posy:
    :param t_height:
    :param t_width:
    :param pad:
    :return:
    '''
    length = annotation.length
    if annotation.length < conf.MIN_CROP:
      length = conf.MIN_CROP  # to prevent really small objects

    center_x = annotation.x - posx * t_width
    center_y = annotation.y - posy * t_height
    tlx = int(max(0, int(center_x - (length / 2) - pad)))
    tly = int(max(0, int(center_y - (length / 2) - pad)))
    brx = int(min(t_width, center_x + (length / 2) + pad))
    bry = int(min(t_height, center_y + (length / 2) + pad))
    return brx, bry, tlx, tly