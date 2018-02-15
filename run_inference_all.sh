#!/bin/bash
# Run inference on all models.  Assumes models are exported using export_
# Execute inference with run_inference.sh <tfrecord> <model> <gpu device> <discard_image_pixels|keep_image_pixels>
# Discard image pixels when benchmarking the performance on your GPU
# e.g. run_inference.sh INFERENCE_TEST.record faster_rcnn_resnet101_coco_300_smallanchor 0 keep_image_pixels
set -x
function tmux_start_inference { tmux new-window -d  -n "inference:GPU$1" "${*:2}; exec bash"; }

models=(
["0"]="faster_rcnn_resnet101_coco_600_smallanchor" \
["1"]="faster_rcnn_resnet101_coco_300_smallanchor" \
["2"]="faster_rcnn_resnet101_coco_100_smallanchor" \
["3"]="faster_rcnn_resnet101_coco_50_smallanchor" \
["4"]="rfcn_resnet101_coco_300_smallanchor" \
["5"]="rfcn_resnet101_coco_100_smallanchor" \
["6"]="rfcn_resnet101_coco_50_smallanchor" \
["7"]="ssd_inception_v2_coco_500" \
)
NUM_MODELS=8

# Run inference across GPU last, using four models at a time
for i in $(seq 0 4 $(($NUM_MODELS-1))); do
  tmux new-session -d -s "inference"
  tmux_start_inference 0 ./run_inference.sh INFERENCE_TEST.record ${models[i]} 2 discard_image_pixels
  tmux_start_inference 1 ./run_inference.sh INFERENCE_TEST.record ${models[i + 1]} 3 discard_image_pixels
  tmux_start_inference 2 ./run_inference.sh INFERENCE_TEST.record ${models[i + 2]} 2 discard_image_pixels
  tmux_start_inference 3 ./run_inference.sh INFERENCE_TEST.record ${models[i + 3]} 3 discard_image_pixels
  sleep 5m
  tmux kill-session -t inference
done
./time_plot.sh
exit

# Run inference each across CPU  last
for i in $(seq 0 1 $(($NUM_MODELS-1))); do
  ./run_inference.sh INFERENCE_TEST.record ${models[i]} -1 discard_image_pixels
done
