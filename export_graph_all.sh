#!/bin/bash
# Export graph of all training models
# This is needed before running inference on images
# usage: export_graph_all.sh
set -x
function tmux_start_export { tmux new-window -d  -n "export:GPU$1" "${*:2}; exec bash"; }
models=(
["0"]="faster_rcnn_resnet101_coco_600_smallanchor" \
["1"]="faster_rcnn_resnet101_coco_300_smallanchor" \
["2"]="faster_rcnn_resnet101_coco_100_smallanchor" \
["3"]="faster_rcnn_resnet101_coco_50_smallanchor"
["4"]="rfcn_resnet101_coco_600_smallanchor" \
["5"]="rfcn_resnet101_coco_300_smallanchor" \
["6"]="rfcn_resnet101_coco_100_smallanchor" \
["7"]="rfcn_resnet101_coco_50_smallanchor" \
["8"]="ssd_inception_v2_coco_500"
)
NUM_MODELS=8
# Export two models at a time
for i in $(seq 0 4 $(($NUM_MODELS-1))); do
tmux new-session -d -s "export"
  tmux_start_export 0 ./export_graph.sh ${models[i]} 0
  tmux_start_export 1 ./export_graph.sh ${models[i + 1]} 1
  tmux_start_export 2 ./export_graph.sh ${models[i + 2]} 2
  tmux_start_export 3 ./export_graph.sh ${models[i + 3]} 3
sleep 2m
tmux kill-session -t export
done
