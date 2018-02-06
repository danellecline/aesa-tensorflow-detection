#!/usr/bin/env bash
# Run inference command on a DevBox
# Execute inference with run_inference.sh <tfrecord> <model> <gpu device>
# e.g. run_inference.sh M56_900x900_test_by_group.record faster_rcnn_resnet101_coco_100_smallanchor 0
source ~/Dropbox/GitHub/venv-aesa-tensorflow-detection-devbox/bin/activate
pushd tensorflow_models/research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim:`pwd`/object_detection
popd
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/
export CUDA_VISIBLE_DEVICES="$3"

TF_RECORD_FILE="$1"
MODEL_DIR=`pwd`//models/export/"$2"/frozen_inference_graph.pb

echo python tensorflow_models/research/object_detection/inference/infer_detections.py \
  --input_tfrecord_paths=$TF_RECORD_FILE \
  --output_tfrecord_path=$TF_RECORD_FILE\.detections.record \
  --inference_graph=$MODEL_DIR
  
#--discard_image_pixels
