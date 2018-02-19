#!/usr/bin/env python
__author__    = 'Danelle Cline'
__copyright__ = '2018'
__license__   = 'GPL v3'
__contact__   = 'dcline at mbari.org'
__doc__ = '''

Used for clearning bad tiles from the data
@var __date__: Date of last svn commit
@undocumented: __doc__ parser
@status: production
@license: GPL
'''
import os
import image_utils
import glob

tile_png_dir = '/Volumes/ScratchDrive/AESA/M56_10_1'
dir = '{0}/data/M56_600x600_by_group/'.format(os.getcwd())
all_files = glob.iglob(dir + "**/*.png", recursive=True)
w = 600
h = 600
for file in all_files:
  if os.stat(file).st_size == 0:
    bname = os.path.basename(file)
    tile = '{0}/{1}'.format(tile_png_dir, bname)
    rescaled_file = '{0}/PNGImages/{1}'.format(dir, file)
    cmd = '/usr/local/bin/convert {0} -scale {1}x{2}\! "{3}"'.format(tile, w, h, rescaled_file)
    print(cmd)
    #os.system(cmd)

'''dir = "/Volumes/ScratchDrive/AESA/M56_images_960_540/"
all_files = glob.iglob(dir + "**/*.png", recursive=True) 
for file in all_files:
  height, width = image_utils.get_dims(file)
  if width > 14000:
    print('Removing {0}'.format(file))
    os.remove(file)
