#!/usr/bin/env bash
source ~/Dropbox/GitHub/venv-aesa-tensorflow-detection/bin/activate
cd ~/Dropbox/GitHub/aesa-tensorflow-detection/
TOP_DIR='/Users/dcline/Dropbox/GitHub/aesa-tensorflow-detection/'
python create_tfrecord.py \
--collection M56 \
--data_dir $TOP_DIR/data/ \
--output_path $TOP_DIR/data/AESA_M56_train.record \
--label_map_path $TOP_DIR/data/aesa_k5_label_map.pbtxt \
--set train

python create_tfrecord.py \
--collection M56 \
--data_dir $TOP_DIR/data/ \
--output_path $TOP_DIR/data/AESA_M56_test.record \
--label_map_path $TOP_DIR/data/aesa_k5_label_map.pbtxt \
--set test