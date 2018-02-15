#!/bin/bash
# Trains all models for 3 hours each, splitting testing/training on separate GPUS
# as needed for memory constraints
# To run with 500x500 tiles:  ./run_all.sh 500
set -x
function tmux_start_train { tmux new-window -d  -n "train:GPU$1" "${*:2}; exec bash"; }
function tmux_start_test { tmux new-window -d  -n "test:GPU$1" "${*:2}; exec bash"; }

OUT_DIR="models_"${1}"x"${1}"_by_group"
mkdir -p $OUT_DIR
find ./models -name pipeline_devbox.config -exec sed -i  "s/.*min_dimension.*/        min_dimension: $1 /" '{}' \; 
find ./models -name pipeline_devbox.config -exec sed -i  "s/.*max_dimension.*/        max_dimension: $1 /" '{}' \;
find ./models -name pipeline_devbox.config -exec sed -i  "s/500x500/$1x$1/g" '{}' \;
find ./models -name pipeline_devbox.config -exec sed -i  "s/1000x1000/$1x$1/g" '{}' \;

models=(
["0"]="ssd_inception_v2_coco_500" \
["1"]="rfcn_resnet101_coco_50_smallanchor" \
["2"]="rfcn_resnet101_coco_100_smallanchor" \
["3"]="rfcn_resnet101_coco_300_smallanchor" \
["4"]="rfcn_resnet101_coco_600_smallanchor" \
["5"]="rfcn_resnet101_coco_100_tinyanchor"
)
NUM_MODELS=6
# Run two models at a time for 3 hours each, splitting testing and training across 4 GPUS
for i in $(seq 0 2 $(($NUM_MODELS-1))); do
tmux new-session -d -s "train"
  tmux_start_train 0 ./run.sh ${models[i]} train 0
  tmux_start_train 1 ./run.sh ${models[i + 1]} train 1
sleep 10s
tmux new-session -d -s "test"
  tmux_start_test 2 ./run.sh ${models[i]} test 2
  tmux_start_test 3 ./run.sh ${models[i + 1]} test 3
sleep 4h
tmux kill-session -t train
tmux kill-session -t test
sleep 10s
mkdir -p $OUT_DIR/${models[i]}/eval
mkdir -p $OUT_DIR/${models[i]}/train
mkdir -p $OUT_DIR/${models[i+1]}/eval
mkdir -p $OUT_DIR/${models[i+1]}/train
cp -Rf models/${models[i]}/eval $OUT_DIR/${models[i]}
cp -Rf models/${models[i]}/train $OUT_DIR/${models[i]}
cp -Rf models/${models[i+1]}/eval $OUT_DIR/${models[i+1]}
cp -Rf models/${models[i+1]}/train $OUT_DIR/${models[i+1]}
#rm -Rf models/${models[i]}/eval
#rm -Rf models/${models[i]}/train
#rm -Rf models/${models[i+1]}/eval
#rm -Rf models/${models[i+1]}/train
done
