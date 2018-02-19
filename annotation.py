#!/usr/bin/env python
__author__ = 'Danelle Cline'
__copyright__ = '2018'
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
import csv
import cv2
import numpy
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
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
  missing_frames = []

  aesa_annotation = namedtuple("Annotation", ["category", "group", "heirarchical_group", "x", "y", "pad",
                                              "filename", "length"])
  image_metadata = namedtuple("Metadata", ["tile_width", "tile_height", "bins_x", "bins_y", "image_width",
                                           "image_height", "rotate"])

  def __init__(self, clean_file = None, group_file = None, heirarchy_file = None):
      print('Parsing ' + conf.ANNOTATION_FILE)
      self.df_annotations = pd.read_csv(conf.ANNOTATION_FILE, sep=',')
      self.csv_file = conf.ANNOTATION_FILE[conf.ANNOTATION_FILE.rfind("/") + 1:]
      self.frame_dict = {}
      crop_index = pd.DataFrame()

      if clean_file:
        print('Parsing ' + clean_file)
        df_clean = pd.read_csv(clean_file, sep=',')
        crop_index = df_clean['CropNo']
        reassigned_category = df_clean['Reassigned.value']

      if heirarchy_file:
          print('Parsing ' + heirarchy_file)
          df = pd.read_csv(heirarchy_file, sep=',')
          for index, row in df.iterrows():
              if np.isnan(float(row['coarse_class'])):
                  print('Category {0} does not have a group'.format(row['category']))
              else:
                  # GROUP1, GROUP2, 1-based indexing
                  print('Grouping category {0} as GROUP{1}'.format(row['category'], int(row['coarse_class']) + 1))
                  self.group_map[row['category']] = 'GROUP{:d}'.format(int(row['coarse_class']) + 1)

      if group_file:
        print('Parsing ' + group_file)
        df = pd.read_csv(group_file, sep=',')
        for index, row in df.iterrows():
          category = row.Category.strip().replace(" ", "")
          group = row.Group.strip().replace(" ", "")
          self.group_map[category.upper()] = group.upper()

      ntest = 0
      for index, row in self.df_annotations.iterrows():
          try:
              f = row['FileName']
              if isinstance(f, str) and '.' in f:
                f = f.replace('.', '')  # handle last . sometimes found Filename column
              if conf.FILE_FORMAT:
                filename = os.path.join(conf.TILE_DIR, conf.FILE_FORMAT % f)
              else:
                filename = os.path.join(conf.TILE_DIR, f)

              #print('Processing row {0} filename {1} annotation {2}'.format(index, filename, row['Category']))

              category = row["Category"].strip().replace(" ", "").upper()
              group = None
              try:
                group = row["group"].strip().replace(" ", "").upper()
              except:
                pass

              if index in crop_index.values:
                idx = np.where(crop_index == index)
                new_category = reassigned_category.iloc[idx[0]].values[0]
                print('Correction index {0} category {1} to {2}'.format(idx[0], category, new_category))
                if new_category == 'Remove':
                  continue
                category = new_category.strip().replace(" ", "").upper()

              x = int(row["CentreX"])
              y = int(row["CentreY"])
              length = float(row["Measurement"])
              if np.isnan(length):
                  length = conf.MIN_CROP
              pad = (length / 100.0) * conf.PAD_PERCENTAGE
              a = self.aesa_annotation(category=category, group=group, heirarchical_group=None, x=x, y=y, pad=pad, filename=filename, length=length)

              if group_file and category not in self.group_map:
                continue

              a = self.correct(a)
              if group_file:
                a = a._replace(group=self.group_map[category])
              if heirarchy_file:
                a = a._replace(heirarchical_group=self.group_map[category])

              self.annotations.append(a)
              #if a.category in conf.OPTIMIZED_CATEGORIES:
              #  self.annotations.append(a)
              #ntest += 1
              #if ntest > 1000:
              #  break;
              #self.annotations.append(a)
              #  break;

          except Exception as ex:
              print("Error processing annotation row {0} filename {1} \n".format(index, filename))


  def aggregate(self):
      '''
        Aggregates each annotation object into a dictionary keyed by frame.
        Requires images to be generated first with generate_tiles().
      :return:
      '''
      csv_file = '{0}_missing.csv'.format(conf.COLLECTION)

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

      a = np.asarray(self.missing_frames)
      a.tofile(csv_file,sep=',',format='%s')


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
              print('Generating tiles for {0}'.format(key))
              if key not in self.frame_dict.keys() and tail not in self.missing_frames :
                  if not os.path.exists(str(a.filename)):
                    self.missing_frames.append(tail)
                    continue
                  image_height, image_width = image_utils.get_dims(a.filename)
                  image_file = '{0}/{1}_{2:02}.png'.format(conf.TILE_PNG_DIR, key, 0)
                  # http://www.imagemagick.org/Usage/crop/#crop_equal
                  if not os.path.exists(image_file):
                    #raise('Skipping converting {0} to tiles due to time constraint'.format(a.filename))
                    print('Converting {0} into tiles'.format(a.filename))
                    if image_height > image_width:
                        os.system('/usr/local/bin/convert "{0}" -crop {1}x{2}@ +repage +adjoin -quality 100%% "{3}/{4}_%02d.png"'
                            .format(a.filename, n_tilesw, n_tilesh, conf.TILE_PNG_DIR, key));
                    else:
                        os.system('/usr/local/bin/convert "{0}" -crop {2}x{1}@ +repage +adjoin -quality 100%% "{3}/{4}_%02d.png"'
                            .format(a.filename, n_tilesw, n_tilesh, conf.TILE_PNG_DIR, key));

                  tile_height, tile_width = image_utils.get_dims(image_file)
                  if image_height > image_width:
                    rotate = False
                  else: # wide mosaic tiles which will be rotated later
                    rotate = True
                  bins_x = numpy.arange(start=0, stop=image_width - tile_width, step=tile_width - 1)
                  bins_y = numpy.arange(start=0, stop=image_height - tile_height, step=tile_height - 1)

                  self.frame_dict[key] = self.image_metadata(tile_width=tile_width, tile_height=tile_height,
                                                         image_width=image_width, image_height=image_height,
                                                         bins_x=bins_x, bins_y=bins_y,  rotate=rotate)

          except Exception as ex:
            print("Error converting image {0}. Exception {1}".format(a.filename, ex))
            self.frame_dict[key] = self.image_metadata(tile_width=-1, tile_height=-1, image_width=-1, image_height=-1,
                                                       bins_x=None, bins_y=None, rotate=False)


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
      if area > 5000 and (x == 0 or w + h > conf.TARGET_TILE_WIDTH/2):
        w_l.append(w)
        x_l.append(x)
        y_l.append(y)
        pt = [float(x), float(x)]
        x_c.append(pt)

    crop_left = 0
    crop_right = 0
    tlx = 0

    # should be two groups: one on left and one on right
    data = np.array(x_c)
    try:
      centroids, _ = kmeans(np.array(x_c), 2, iter=20)
      idx, _ = vq(data, centroids)
      centroids_x = centroids[:,0]

      if len(centroids_x) == 2 :
        group = 0
        distance = np.diff(sorted(centroids_x))
        if int(distance) > conf.TARGET_TILE_WIDTH/2:
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

        if len(centroids_x) == 1:
          x_1 = x_ll[0]
          width = w_ll[0]
          if x_1 == 0:
            crop_left = width
            tlx = crop_left
          else:
            crop_right = width

        tly = 0
        w = tile_width - crop_left - crop_right
        h = tile_height
        cmd = '/usr/local/bin/convert "{0}" -crop {1}x{2}+{3}+{4} +repage "{5}"'.format(tile_file, w, h, tlx, tly, dst_file);
        print('Running {0}'.format(cmd))
        os.system(cmd)
        return crop_left, crop_right
    except Exception as ex:
      print('Error finding black blobs {0}'.format(ex))
      shutil.copyfile(tile_file, dst_file)
      return 0, 0

  def show_annotation(self, title, category, image_file, brx, bry, tlx, tly, write=False, filename=None):
    '''

    :param category:
    :param image_file:
    :param brx:
    :param bry:
    :param tlx:
    :param tly:
    :return:
    '''
    if conf.SHOW_ANNOTATION or write:
      img = cv2.imread(image_file)
      cv2.rectangle(img, (tlx, tly), (brx, bry), (0, 255, 0), 3)
      font = cv2.FONT_HERSHEY_SIMPLEX
      cv2.putText(img, category, (tlx, tly), font, 2, (255, 255, 255), 2)
      cv2.namedWindow(title, cv2.WINDOW_NORMAL)
      width, height = image_utils.get_dims(image_file)
      cv2.resizeWindow(title, int(width/4), int(height/4))
      if conf.SHOW_ANNOTATION:
        cv2.imshow(title, img)
        cv2.waitKey(1000)
      if write:
        cv2.imwrite(filename, img)

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

    src_file = '{0}/{1}_{2:02}.png'.format(conf.TILE_PNG_DIR, key, index)
    cropped_file = '{0}/{1}_{2:02}_cropped.png'.format(conf.TILE_PNG_DIR, key, index)

    brx, bry, tlx, tly, center_x, center_y = self.convert_coords(metadata, annotation, bin_x, bin_y)
    #self.show_annotation('raw', annotation.group, src_file, brx, bry, tlx, tly)

    crop_coord = '{0}/{1}_{2:02}_cropped.csv'.format(conf.TILE_PNG_DIR, key, index)
    if os.path.exists(crop_coord):
        print('Loading {0}'.format(crop_coord))
        with open(crop_coord, 'r') as in_file:
          reader = csv.reader(in_file)
          row = next(reader)
          crop_left=int(row[0])
          crop_right=int(row[1])
    else:
      crop_left, crop_right = self.rotate_crop(annotation, bin_x, bin_y, brx, bry,
                                             cropped_file, metadata,  src_file, tlx, tly)

    with open(crop_coord, 'w') as out:
      csv_out = csv.writer(out)
      csv_out.writerow([crop_left, crop_right])

    if metadata.rotate:
      tile_width = metadata.tile_height
      tile_height = metadata.tile_width
    else:
      tile_height = metadata.tile_height
      tile_width = metadata.tile_width

    # throw annotations out if they fall outside the crop
    if tlx > tile_width - crop_right or tlx < crop_left:
      raise Exception('{0} tlx {1} falls outside of cropped image bounds {2} {3}'.format(annotation.category, tlx,
                                                                                         tile_width,
                                                                                         tile_width - crop_right,
                                                                                         tile_width + crop_left))

    # optimize if needed; not currently working
    '''if conf.OPTIMIZE_BOX == True and (brx - tlx) > conf.MIN_CROP and (bry - tly) > conf.MIN_CROP and \
            annotation.category in conf.OPTIMIZED_CATEGORIES or annotation.group in conf.OPTIMIZED_GROUPS:
        brx, bry, tlx, tly, center_x, center_y = self.optimize(annotation, brx, bry, cropped_file, tlx, tly)'''

    # adjust annotation to cropped tile
    tile_width -= crop_left + crop_right
    tlx -= crop_left
    tlx = min(tlx, tile_width)
    brx -= crop_left
    brx = min(brx, tile_width)
    center_x -= crop_left

    #self.show_annotation('cropped', annotation.group, cropped_file, brx, bry, tlx, tly)
    annotation_new = annotation._replace(x=center_x, y=center_y)

    tile_width2 = int(tile_width/2)
    tile_height2 = int(tile_height/2)
    bins_x2 = numpy.arange(start=0, stop=tile_width - tile_width2, step=tile_width2 - 1)
    bins_y2 = numpy.arange(start=0, stop=tile_height - tile_height2, step=tile_height2 - 1)
    bin_x2 = int(numpy.digitize(center_x, bins_x2)) - 1
    bin_y2 = int(numpy.digitize(center_y, bins_y2)) - 1
    # calculate what sub image index the annotations should fall into
    subindex = bin_y2*2 + bin_x2
    rescaled_file = '{0}/{1}_{2:02}_{3:02}.png'.format(conf.PNG_DIR, key, index, subindex)
    if not os.path.exists(rescaled_file):
      w = tile_width2
      h = tile_height2
      tlxx = bin_x2*w
      tlyy = bin_y2*h
      os.system('/usr/local/bin/convert "{0}" -crop {1}x{2}+{3}+{4} +repage -scale {5}x{6}\! "{7}"'.format(
        cropped_file, w, h, tlxx, tlyy, conf.TARGET_TILE_WIDTH, conf.TARGET_TILE_HEIGHT, rescaled_file))

    sub_metadata = self.image_metadata(tile_width=tile_width2, tile_height=tile_height2,
                        image_width=tile_height2, image_height=tile_width2,
                        bins_x=bins_x2, bins_y=bins_y2, rotate=False)

    # Convert coordinates to the subindex file
    brx2, bry2, tlx2, tly2, center_x2, center_y2 = self.convert_coords(sub_metadata, annotation_new, bin_x2, bin_y2, False)

    # Calculate scaled coordinates
    obj = {}
    scale_x = conf.TARGET_TILE_WIDTH/tile_width2
    scale_y = conf.TARGET_TILE_HEIGHT/tile_height2
    tlx2  = min(int(scale_x*tlx2), conf.TARGET_TILE_WIDTH-1)
    tly2 = min(int(scale_y*tly2), conf.TARGET_TILE_HEIGHT-1)
    brx2 = min(int(scale_x*brx2), conf.TARGET_TILE_WIDTH-1)
    bry2 = min(int(scale_y*bry2), conf.TARGET_TILE_HEIGHT-1)

    # store in dictionary to dump to xml
    if annotation.heirarchical_group:
      obj['name'] = annotation.heirarchical_group
    if annotation.group:
      obj['name'] = annotation.group
    else:
      obj['name'] = annotation.category
    obj['difficult'] = conf.DIFFICULT
    obj['truncated'] = 0
    obj['pose'] = conf.POSE
    obj['bndbox'] = {}
    obj['bndbox']['xmin'] = tlx2
    obj['bndbox']['ymin'] = tly2
    obj['bndbox']['xmax'] = brx2
    obj['bndbox']['ymax'] = bry2

    if tlx2 == brx2 or tly2 == bry2:
      raise Exception("Object too small. Annotation {0}".format(annotation.filename))

    if os.path.exists(rescaled_file):
      if metadata.rotate:
        annotated_file = '{0}/{1}_{2:02}_{3:02}_annotated_r.png'.format(conf.PNG_DIR, key, index, subindex)
      else:
        annotated_file = '{0}/{1}_{2:02}_{3:02}_annotated.png'.format(conf.PNG_DIR, key, index, subindex)
      self.show_annotation('final', annotation.group, rescaled_file, brx2, bry2, tlx2, tly2, True, annotated_file)
    return obj, rescaled_file

  def rotate_crop(self, annotation, bin_x, bin_y, brx, bry, cropped_file, metadata, src_file,
                        tlx, tly):

    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, 'temp.png')

    if metadata.rotate:
      cmd = '/usr/local/bin/convert {0} -rotate 90 "{1}"'.format(src_file, temp_file)
      os.system(cmd)
      print('Running {0}'.format(cmd))
      brx, bry, tlx, tly, center_x, center_y = self.convert_coords(metadata, annotation, bin_x, bin_y, True)
      #self.show_annotation('rotated', annotation.group, temp_file, brx, bry, tlx, tly)
    else:
      shutil.copyfile(src_file, temp_file)
    # crop and store left/right offsets; TODO: add top/bottom crop
    img = cv2.imread(temp_file)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    invert_img = cv2.bitwise_not(binary_img);
    if metadata.rotate:
      crop_left, crop_right = self.crop(invert_img, temp_file, metadata.tile_height, metadata.tile_width, cropped_file)
    else:
      crop_left, crop_right = self.crop(invert_img, temp_file, metadata.tile_width, metadata.tile_height, cropped_file)

    os.removedirs(temp_dir)
    return crop_left, crop_right

  def scale(self, cropped_file, rescaled_file):
    # convert cropped image to the model size
    cmd = '/usr/local/bin/convert {0} -scale {1}x{2}\! "{3}"'.format(cropped_file, conf.TARGET_TILE_WIDTH,
                                                                     conf.TARGET_TILE_HEIGHT, rescaled_file)
    print('Running {0}'.format(cmd))
    os.system(cmd)

  def optimize(self, annotation, brx, bry, src_crop, tlx, tly):

    kernel3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = cv2.imread(src_crop)
    crop_img = img[tly:bry, tlx:brx]
    gray_image = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    cv2.destroyAllWindows()
    ret2, th = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    clean = cv2.erode(th, kernel3, iterations=3)
    # cv2.imshow('Otsu', clean)
    # cv2.imshow('Cropped', crop_img)
    # cv2.waitKey(2000)
    found, x, y, w, h = image_utils.find_object(clean, crop_img, annotation.length * annotation.length)

    if found:
      # adjust the actual bounding box with new localized bounding box coordinates
      tlx += x;
      tly += y;
      brx = tlx + w;
      bry = tly + h;
      cv2.rectangle(img, (tlx, tly), (brx, bry), (255, 0, 0), 3)
    return brx, bry, tlx, tly

  def rotate(self, origin, point, angle):
    '''
    Rotate point counter-clockwise by angle around origin
    :param origin:
    :param point:
    :param angle:
    :return:
    '''
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

  def convert_coords(self, metadata, annotation, posx, posy, rotate=False):
    '''
    Converts coordinates from annotation to tile coordinates
    :param metadata:
    :param annotation:
    :param posx:
    :param posy:
    :return:
    '''
    length = annotation.length
    if annotation.length < conf.MIN_CROP:
      length = conf.MIN_CROP  # to prevent really small objects

    if rotate:
      origin = (0, 0)
      x = annotation.x - posx * metadata.tile_width
      y = annotation.y - posy * metadata.tile_height
      (xx, yy) = self.rotate(origin, (x,y), math.radians(-90))
      # convert to image coordinates
      center_x = metadata.tile_height - xx
      center_y = -1*yy
      tlx = int(max(0, int(center_x - (length / 2) - annotation.pad)))
      tly = int(max(0, int(center_y - (length / 2) - annotation.pad)))
      brx = int(min(metadata.tile_height, center_x + (length / 2) + annotation.pad))
      bry = int(min(metadata.tile_width, center_y + (length / 2) + annotation.pad))
    else:
      w = metadata.tile_width
      h = metadata.tile_height
      center_x = annotation.x - posx * metadata.tile_width
      center_y = annotation.y - posy * metadata.tile_height

      tlx = int(max(0, int(center_x - (length / 2) - annotation.pad)))
      tly = int(max(0, int(center_y - (length / 2) - annotation.pad)))
      brx = int(min(w, center_x + (length / 2) + annotation.pad))
      bry = int(min(h, center_y + (length / 2) + annotation.pad))

    return brx, bry, tlx, tly, center_x, center_y
