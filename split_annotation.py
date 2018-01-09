import glob
import os
from bs4 import BeautifulSoup, Tag
import conf
import random

if __name__ == '__main__':

    annotations = []
    xml_in = []
    for xml_in in glob.iglob(confANNOTATION_DIR + '*.xml', recursive=False):
        print('Found {0}'.format(xml_in))
        # find the source image in the xml_in file and replace
        infile = open(xml_in, "r")
        contents = infile.read()
        soup = BeautifulSoup(contents, 'xml')

        # skip over annotations with no objects
        if not soup.find('object'):
            print('No objects found in {0} so skipping'.format(xml_in))
            continue

        # otherwise keep in list
        path, filename = os.path.split(xml_in)
        annotations.append(filename)

    # split randomly in to 80% train 20% test; note that this may not give an even split on
    # all classes, as it's split by filename and not the objects in the frame; TODO: revisit train/test split
    # by class
    random.shuffle(annotations)

    total = len(annotations)
    train_data = annotations[:round(0.8*total)]
    test_data = annotations[round(0.2*total):]

    with open(os.path.join(DATA_DIR, 'train.txt'), 'w') as f:
        for l in train_data:
            path, filename = os.path.split(l)
            f.write(filename + '\n')

    with open(os.path.join(DATA_DIR, 'test.txt'), 'w') as f:
        for l in test_data:
            path, filename = os.path.split(l)
            f.write(l + '\n')

    print('Done')




