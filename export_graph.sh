#!/usr/bin/env bash
# Export graph of training model
# This is needed before running inference on images
# Finds the latest checkpoint and exports that by default 
# usage: export_graph.sh <modelname> <GPU>
# e.g. export_graph.sh faster_rcnn_resnet101_coco_20_smallanchor 0
set -x
source $TF_VENV/bin/activate
pushd tensorflow_models/research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim:`pwd`/object_detection
popd
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/
export CUDA_VISIBLE_DEVICES="$2"

OUT_DIR=models/export/"$1"
IN_DIR=models/"$1"/train
if [ -d "$OUT_DIR" ]; then
  rm -Rf $OUT_DIR
fi
mkdir -p $OUT_DIR 
last_checkpoint="$(find $IN_DIR -name '*ckpt*' | sort | head -n 1)"
string2="$(cut -d- -f2 <<< $last_checkpoint)"
prefix="$(cut -d. -f1 <<< $string2)"

python tensorflow_models/research/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path models/"$1"/pipeline_devbox.config \
    --trained_checkpoint_prefix models/"$1"/train/model.ckpt-"$prefix" \
    --output_directory $OUT_DIR
