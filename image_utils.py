#!/usr/bin/env python

__author__    = 'Danelle Cline'
__copyright__ = '2018'
__license__   = 'GPL v3'
__contact__   = 'dcline at mbari.org'
__doc__ = '''

Utility class for image processing methods 

@var __date__: Date of last svn commit
@undocumented: __doc__ parser
@status: production
@license: GPL
'''

import os
import re
import subprocess
import cv2

def find_object(image_bin, image_color, max_area):
  """
  find object blob and return its coordinates
  :param image: the image in opencv format
  :param max_area: maximum area of blob
  :return: x, y, w, h in the input image coordinates
  """
  invert = cv2.bitwise_not(image_bin);
  #cv2.imshow('invert', invert)
  #cv2.waitKey(500)

  # get blobs
  im, contours, heirachy = cv2.findContours(invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  img = cv2.drawContours(invert, contours, -1, (0, 255, 0), 7)
  #cv2.imshow('contours', img)
  #cv2.waitKey(500)

  # keep valid contours
  valid_contours = []
  cnt = 0
  for c in contours:
    try:
      # get rectangle bounding contour
      [x, y, w, h] = cv2.boundingRect(c)
      area = cv2.contourArea(c)
      #print("area {0} x {1} y {2} w {3} h {4}".format(area, x, y, w, h))
      # get valid areas, not small blobs
      if area > 2000 and area < max_area :
        pt = [float(y), float(y)]
        cn = contours[cnt]
        img = cv2.drawContours(image_color, [cn], 0, (255, 0, 0), 5)
        #cv2.imshow('Possible', img)
        #cv2.waitKey(500)
        valid_contours.append(c)
      cnt += 1

    except Exception as ex:
      print(ex)

  if len(valid_contours) > 0:
    # find largest contour in mask
    c = max(valid_contours, key=cv2.contourArea)
    [x, y, w, h] = cv2.boundingRect(c)
    return True, x, y, w, h

  return False, -1, -1, -1, -1


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


def write_annotation(annot_dict, image_file, out_dir):
    '''
     Simple utility to save the annotation box overlayed on an image
    :param annot_dict:
    :param image_file:
    :param out_dir:
    :return:
    '''
    img = cv2.imread(image_file)
    filename = annot_dict['filename']
    for obj in annot_dict['object']:
      name = obj['name']
      tlx = int(obj['bndbox']['xmin'])
      tly = int(obj['bndbox']['ymin'])
      brx = int(obj['bndbox']['xmax'])
      bry = int(obj['bndbox']['ymax'])
      cv2.rectangle(img, (tlx, tly), (brx, bry), (0, 255, 0), 3)
      font = cv2.FONT_HERSHEY_SIMPLEX
      cv2.putText(img, name, (tlx, tly), font, 1, (255, 255, 255), 2)
    root, ext = os.path.splitext(filename)
    cv2.imwrite(os.path.join(out_dir, '{0}_a{1}'.format(root, ext)), img)