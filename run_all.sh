#!/bin/bash
set -x
function tmux_start_train { tmux new-window -d  -n "train:GPU$1" "${*:2}; exec bash"; }
function tmux_start_test { tmux new-window -d  -n "test:GPU$1" "${*:2}; exec bash"; }

models=(
["0"]="faster_rcnn_resnet101_coco_300_smallanchor" \
["1"]="faster_rcnn_resnet101_coco_100_smallanchor" \
["2"]="faster_rcnn_resnet101_coco_50_smallanchor" \
["3"]="rfcn_resnet101_coco_300_smallanchor" \
["4"]="rfcn_resnet101_coco_100_smallanchor" \
["5"]="rfcn_resnet101_coco_50_smallanchor")

NUM_MODELS=6
# Run two models at a time for 3 hours each, splitting testing and training across 4 GPUS
for i in $(seq 0 2 $(($NUM_MODELS-1))); do
tmux new-session -d -s "train"
  tmux_start_train 0 ./run.sh ${models[i]} train 0
  tmux_start_train 1 ./run.sh ${models[i + 1]} train 1

tmux new-session -d -s "test"
  tmux_start_test 2 ./run.sh ${models[i]} test 2
  tmux_start_test 3 ./run.sh ${models[i + 1]} test 3
sleep 3h
tmux kill-session -t train
tmux kill-session -t test
done
