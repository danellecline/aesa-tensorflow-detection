#!/usr/bin/env bash
source $TF_VENV/bin/activate 
python create_tfrecord.py \
--collection M56_1000x1000_by_group M535455_1000x1000_by_group \
--data_dir $PWD/data/ \
--output_path $pWD/data/M53545556_1000x1000_train_by_group.record \
--label_map_path $PWD/data/aesa_group_map.pbtxt \
--set train

python create_tfrecord.py \
--collection M56_1000x1000_by_group M535455_1000x1000_by_group \
--data_dir $PWD/data/ \
--output_path $PWD/data/M53545556_1000x1000_test_by_group.record \
--label_map_path $PWD/data/aesa_group_map.pbtxt \
--set test
