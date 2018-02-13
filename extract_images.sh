#!/usr/bin/env bash
BASE_DIR=~/Dropbox/GitHub/mbari-tensorflow-detection 
source ~/Dropbox/GitHub/venv-aesa-tensorflow-detection-devbox/bin/activate
RECORD=`pwd`/data/imagerecords/$1
OUT_PATH=`pwd`/data/$1-dump/
LABEL_PATH=`pwd`/data/mbari_benthic_label_subset_map.pbtxt
mkdir -p $OUT_PATH
python extract_images.py -r $RECORD -o $OUT_PATH -l $LABEL_PATH
