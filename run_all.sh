#!/bin/bash
set -x  
function tmux_start_train { tmux new-window -d  -n "train:GPU$1" "${*:2}; exec bash"; }
function tmux_start_test { tmux new-window -d  -n "test:CPU$1" "${*:2}; exec bash"; }
IDX=$(seq 0 3)
models=( ["0"]="faster_rcnn_resnet101_coco_100" ["1"]="faster_rcnn_resnet101_coco_300" \
	 ["2"]="faster_rcnn_resnet101_coco_20" ["3"]="faster_rcnn_resnet101_coco_50" )
tmux new-session -d -s "train" 
for gpu_index in $IDX; do
  tmux_start_train ${gpu_index} \
  ./run.sh ${models[gpu_index]} train ${gpu_index} 
done
tmux new-session -d -s "test" 
for index in $IDX; do
  tmux_start_test ${index} \
  ./run.sh ${models[index]} test
done
sleep 12h
tmux kill-session -t train
tmux kill-session -t test
