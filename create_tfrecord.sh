#!/usr/bin/env bash
source ~/Dropbox/GitHub/venv-aesa-tensorflow-detection/bin/activate
cd ~/Dropbox/GitHub/aesa-tensorflow-detection/
TOP_DIR='/Users/dcline/Dropbox/GitHub/aesa-tensorflow-detection/'
python create_tfrecord.py \
--collection M56_1000x1000_by_group M535455_1000x1000_by_group \
--data_dir $TOP_DIR/data/ \
--output_path $TOP_DIR/data/M53545556_1000x1000_train_by_group.record \
--label_map_path $TOP_DIR/data/aesa_group_map.pbtxt \
--set train

python create_tfrecord.py \
--collection M56_1000x1000_by_group M535455_1000x1000_by_group \
--data_dir $TOP_DIR/data/ \
--output_path $TOP_DIR/data/M53545556_1000x1000_test_by_group.record \
--label_map_path $TOP_DIR/data/aesa_group_map.pbtxt \
--set test
