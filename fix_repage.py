import os
import shutil
import glob
import subprocess
import re

tile_png_dir = '/Volumes/ScratchDrive/AESA/M535455_10_1'

def fix_repage(image):
  """
  get the height and width and crop of a tile
  :param image: the image file
  :return: height, width, crop
  """
  cmd = '/usr/local/bin/identify "{0}"'.format(image)
  subproc = subprocess.Popen(cmd, env=os.environ, shell=True, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                             stdout=subprocess.PIPE)
  out, err = subproc.communicate()
  # get image height and width of raw tile
  p = re.compile(r'(?P<c_width>\d+)x(?P<c_height>\d+)')
  match = re.search(pattern=p, string=str(out))
  if (match):
    c_width = int(match.group("c_width"))
    c_height = int(match.group("c_height"))

  p = re.compile(r'(?P<t_width>\d+)x(?P<t_height>\d+)\+(?P<left_crop>\d+)\+(?P<right_crop>\d+)')
  match = re.search(pattern=p, string=str(out))
  if (match):
    left_crop = int(match.group("left_crop"))
    if left_crop != 0:
      print('Fixing repage on {0} left crop{1}'.format(image, left_crop))
      cmd = '/usr/local/bin/convert {0} +repage /tmp/crop.png'.format(image)
      os.system(cmd)
      shutil.copyfile('/tmp/crop.png', image)

if __name__ == '__main__':
  all_files = glob.iglob(tile_png_dir + "**/*_cropped.png", recursive=True)
  for file in all_files:
    fix_repage(file)