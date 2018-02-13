#!/bin/bash
set -x
function tmux_start_inference { tmux new-window -d  -n "inference:GPU$1" "${*:2}; exec bash"; }

models=(
["0"]="faster_rcnn_resnet101_coco_600_smallanchor" \
["1"]="faster_rcnn_resnet101_coco_300_smallanchor" \
["2"]="faster_rcnn_resnet101_coco_100_smallanchor" \
["3"]="faster_rcnn_resnet101_coco_50_smallanchor"
)
#["5"]="rfcn_resnet101_coco_300_smallanchor" \
#["4"]="rfcn_resnet101_coco_100_smallanchor" \
#["5"]="rfcn_resnet101_coco_50_smallanchor" \
#["6"]="ssd_inception_v2_coco_500" \
#)
NUM_MODELS=4

# Run inference across GPU last, using four models at a time
for i in $(seq 0 4 $(($NUM_MODELS-1))); do
  tmux new-session -d -s "inference"
  tmux_start_inference 0 ./run_inference.sh INFERENCE_TEST.record ${models[i]} 0 discard_image_pixels
  tmux_start_inference 1 ./run_inference.sh INFERENCE_TEST.record ${models[i + 1]} 1 discard_image_pixels
  tmux_start_inference 2 ./run_inference.sh INFERENCE_TEST.record ${models[i + 2]} 2 discard_image_pixels
  tmux_start_inference 3 ./run_inference.sh INFERENCE_TEST.record ${models[i + 3]} 3 discard_image_pixels
  sleep 2m
  tmux kill-session -t inference
done

# Run inference each across CPU  last
for i in $(seq 0 1 $(($NUM_MODELS-1))); do
  ./run_inference.sh INFERENCE_TEST.record ${models[i]} -1 discard_image_pixels
done
