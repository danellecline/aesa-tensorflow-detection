import os
import image_utils
import glob
dir = "/Volumes/ScratchDrive/AESA/M56_images_960_540/"
all_files = glob.iglob(dir + "**/*.png", recursive=True)

for file in all_files:
  height, width = image_utils.get_dims(file)
  if width > 14000:
    print('Removing {0}'.format(file))
    os.remove(file)