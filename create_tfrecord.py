#!/usr/bin/env python
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

from matplotlib import pyplot as plt
import os 
import math
import numpy
import pandas as pd
import sys
from collections import namedtuple 
import argparse
import subprocess
import util
import cv2
import numpy as np

def process_command_line():
    from argparse import RawTextHelpFormatter

    examples = 'Examples:' + '\n\n'
    examples += sys.argv[0] + """--in_dir /Volumes/ScratchDrive/AESA/M56 tiles/raw/ --by_category
              --out_dir /Users/dcline/Dropbox/GitHub/aesa-tensorflow-detection/data/cropped_images
              --annotation_file /Volumes/ScratchDrive/AESA/M56_Annotations_v10.csv \n"""
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
                                     description='Extract cropped images from tiles and associated annotations',
                                     epilog=examples)
    group = parser.add_mutually_exclusive_group()
    parser.add_argument('--in_dir', type=str, required=True, default='/Volumes/ScratchDrive/AESA/M535455_tiles/', help="Path to folders of raw tiled images.")
    parser.add_argument('--out_dir', type=str, required=False, default=os.path.join(os.getcwd(), 'data','cropped_images'), help="Path to store cropped images.")
    parser.add_argument('--annotation_file', type=str, required=True, default='/Volumes/ScratchDrive/AESA/M56_Annotations_v10.csv', help="Path to annotation file.")
    parser.add_argument('--file_format', default='M56_10441297_', type=str, required=False, help="Alternative file prefix to use for calculating the associated frame the annotation is from, e.g. M56_10441297_%d.jpg'")
    parser.add_argument('--strip_trailing_dot', action='store_true', required=False, help="Strip trailing .'")
    args = parser.parse_args()
    return args

def get_dims(image):
    # get the height and width of a tile
    cmd = 'identify %s' % (image)
    subproc = subprocess.Popen(cmd, env=os.environ, shell=True, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                               stdout=subprocess.PIPE)
    out, err = subproc.communicate()
    f = out.rstrip()
    a = f.split(' ')[3]
    size = a.split('+')[0]
    width = int(size.split('x')[0])  # /4
    height = int(size.split('x')[1])  # /4
    return height, width

# finds object
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
     Convert annotation to tensorflow record
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

    image_file = '{0}/{1}_{2:02}.jpg'.format(img_dir, stem, index)
 
    img = cv2.imread(image_file)
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
        
    crop_img = img[tly:tly + int(crop_pixels), tlx:tlx+int(crop_pixels)] 
    gray_image = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY) 
    display_annotation(annotation, brx, bry, img, tlx, tly)
    
    ret2, th = cv2.threshold(gray_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2))   
    kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    erode = cv2.erode(th,kernel,iterations = 1)
    clean = cv2.dilate(erode,kernel2,iterations = 1)

    '''blur_img = cv2.medianBlur(gray_image, 5) 
th1 = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                            cv2.THRESH_BINARY, 11, 2)
th2 = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                            cv2.THRESH_BINARY, 11, 2)

titles = ['Original Image',  'Otsu', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding'] 

images = [crop_img, clean, th1, th2]
for i in range(4):
    plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
#plt.show()'''
    
    # first try Otsu
    cv2.imshow('Otsu', clean)
    found, x, y, w, h =  find_object(clean, crop_img)
    
    if 'Cnidaria' not in a.category and 'Ophiur' not in a.category: 
        if not found: 
            # Next try threshold in increments of 2 
            for thresh in range(3, 13, 2):
                th = cv2.adaptiveThreshold(th, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                            cv2.THRESH_BINARY, thresh, 2)
                clean = cv2.erode(th, kernel, iterations=1)
                clean = cv2.dilate(clean, kernel2, iterations=2)
                clean = cv2.erode(clean, kernel, iterations=1)
                clean = cv2.dilate(clean, kernel2, iterations=2)
        
                cv2.imshow('Thresh {0}'.format(thresh), th)
                found, x, y, w, h = find_object(th, crop_img)
                if found:
                    break;
         
        if found:
            # adjust the actual bounding box
            tlx += x; tly += y; brx = tlx + w; bry = tly + h;
            display_annotation(annotation, brx, bry, img, tlx, tly)
        
        
    print('done')
    cv2.destroyAllWindows() 


def display_annotation(annotation, brx, bry, img, tlx, tly):
    cv2.rectangle(img, (tlx, tly), (brx, bry), (0, 255, 0), 3)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, annotation.category, (tlx, tly), font, 2, (255, 255, 255), 2)
    cv2.imshow("Annotation", img) 


if __name__ == '__main__':
  args = process_command_line()

  util.ensure_dir(args.out_dir)
  failed_file = open(os.path.join(args.out_dir, 'failed_crops.txt'), 'w')

  aesa_annotation = namedtuple("Annotation", ["category", "centerx", "centery", "mtype", "measurement", "index"])

  try:
    print('Parsing ' + args.annotation_file)
    df = pd.read_csv(args.annotation_file, sep=',')
    width = None
    height = None
    TARGET_TILE_WIDTH = 960
    TARGET_TILE_HEIGHT = 540
    n_tilesh = None
    n_tilesw = None
    actual_width = None
    actual_height = None
    converted = {}

    util.ensure_dir(args.out_dir)

    for index, row in df.iterrows():

      try:
        f = row['FileName'] 
        if args.file_format:
          filename = os.path.join(args.in_dir, '{0}{1}.jpg'.format(args.file_format, f))
        else:
          filename = os.path.join(args.in_dir, f)

        if not os.path.exists(filename):
            failed_file.writelines(filename)
            continue

        # get image height and width of raw tile; only do this once assuming all times are the same
        if not width and not height:
            height, width = util.get_dims(filename)
            # create tiled grid closest to 960x540
            n_tilesh = math.ceil(height / TARGET_TILE_HEIGHT)
            n_tilesw = math.ceil(width / TARGET_TILE_WIDTH)

        if filename not in converted.keys():
            converted[filename] = 1
            head, tail = os.path.split(filename)
            stem = tail.split('.')[0]
            # http://www.imagemagick.org/Usage/crop/#crop_equal
            os.system('/usr/local/bin/convert "{0}" -crop {1}x{2}@ +repage +adjoin -quality 100%% "{3}/{4}_%02d.jpg"'.
                  format(filename, n_tilesw, n_tilesh, args.in_dir, stem));
            image_file = '{0}/{1}_{2:02}.jpg'.format(args.in_dir, stem, 0)

            if not actual_height and not actual_width:
                actual_height, actual_width = util.get_dims(image_file)

        print('Processing row {0} filename {1} annotation {2}'.format(index, filename, row['Category']))
        a = aesa_annotation(category=row['Category'],centerx=row['CentreX'], centery=row['CentreY'],
                            measurement=row['Measurement'], mtype=row['Type'],
                            index=index)
        convert_annotation(filename, args.in_dir, a, height, width, actual_height, actual_width)

      except Exception as ex:
          failed_file.write("Error cropping annotation row {0} filename {1} \n".format(index, filename))

  except Exception as ex:
      print(ex)

  print('Done')

