#!bin/env bash
# Run inference command on exported model 
# Requires all models be exported previously with export_graphs_all.sh
set -x
source $TF_VENV/bin/activate
pushd tensorflow_models/research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim:`pwd`/object_detection
popd
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/
if [ "$3" == -1 ]
then
export CUDA_VISIBLE_DEVICES=""
CSV_FILE=`pwd`/data/"$1"-"$2"-CPU-time.csv
else
export CUDA_VISIBLE_DEVICES="$3"
CSV_FILE=`pwd`/data/"$1"-"$2"-GPU-time.csv
fi
TF_RECORD_FILE_IN=`pwd`/data/"$1"
MODEL_DIR=`pwd`/models/export/"$2"/frozen_inference_graph.pb
mkdir -p `pwd`/data/imagerecords/

if [ "$4" == "discard_image_pixels" ]; then
TF_RECORD_FILE_OUT=`pwd`/data/imagerecords/"$1"-"$2"_detections.record
python tensorflow_models/research/object_detection/inference/infer_detections.py \
  --input_tfrecord_paths=$TF_RECORD_FILE_IN \
  --output_tfrecord_path=$TF_RECORD_FILE_OUT \
  --csv_path=$CSV_FILE \
  --inference_graph=$MODEL_DIR \
  --discard_image_pixels
else
TF_RECORD_FILE_OUT=`pwd`/data/imagerecords/"$1"-"$2"_detections_images.record
python tensorflow_models/research/object_detection/inference/infer_detections.py \
  --input_tfrecord_paths=$TF_RECORD_FILE_IN \
  --output_tfrecord_path=$TF_RECORD_FILE_OUT \
  --csv_path=$CSV_FILE \
  --inference_graph=$MODEL_DIR
fi
