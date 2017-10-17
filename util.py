#!/usr/bin/env python
__author__    = 'Danelle Cline'
__copyright__ = '2017'
__license__   = 'GPL v3'
__contact__   = 'dcline at mbari.org'
__doc__ = '''

Utility class  

@var __date__: Date of last svn commit
@undocumented: __doc__ parser
@status: production
@license: GPL
'''
 
import os 
import subprocess 
import re
 
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

def ensure_dir(d):
  """
  ensures a directory exists; creates it if it does not
  :param fname:
  :return:
  """
  if not os.path.exists(d):
    os.makedirs(d)

def crop(tmpdir, image_path):
  path, filename = os.path.split(image_path)
  thumbnail_file = os.path.join(tmpdir, 'thumb_' + filename)
  os.system('convert "%s" -thumbnail 50x50 -unsharp 0x.5 "%s"' % (image_path, thumbnail_file))
  return thumbnail_file
