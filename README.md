# aesa-tensorflow-detection 
 
Code for testing Tensorflow detection API on on AESA data

## Prerequisites
 
- Python version  3.5
- [Protobuf](https://developers.google.com/protocol-buffers/
- Packages: python-tk e.g. apt-get install python3.5-tk,  

## Running

### Check-out the code

    $ git clone https://github.com/danellecline/aesa-tensorflow-detection

### Create virtual environment with correct dependencies

    $ cd aesa-tensorflow-detection
    $ pip3 install virtualenv
    $ virtualenv --python=/usr/local/bin/python3.5 venv-aesa-tensorflow-detection
    $ source venv-aesa-tensorflow-detection/bin/activate
    $ pip3 install -r requirements.txt
    
### Install Tensorflow for Mac OSX
    $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.4.0-py3-none-any.whl
    $ pip3 install --upgrade $TF_BINARY_URL
    
### Install Tensorflow for Ubuntu GPU
Also see [https://www.tensorflow.org/install/install_linux](https://www.tensorflow.org/install/install_linux)

    $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0-cp35-cp35m-linux_x86_64.whl 
    $ pip3 install --upgrade tensorflow-gpu==1.4.0
    
### Install Tensorflow models and object detection protocols

``` bash
git clone https://github.com/tensorflow/models.git tensorflow_models
push tensorflow_models/research/  
#  Download protoc version 3.3 (already compiled). 
mkdir protoc_3.3
cd protoc_3.3
wget https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-linux-x86_64.zip
chmod 775 protoc-3.3.0-linux-x86_64.zip
unzip protoc-3.3.0-linux-x86_64.zip 
protoc object_detection/protos/*.proto --python_out=.
popd 
```

### Add libraries to PYTHONPATH

When running locally, the tensorflow_models directories should be appended to PYTHONPATH. 
This can be done by running the following from tensorflow_models :

``` bash
pushd tensorflow_models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
popd
```

### Generate the TFRecord files 
``` bash
wget URL_FOR_TRAINING_DATA
python create_tfrecord.py  
    --data_dir PATH_TO_TRAINING_DATA --collection aesa_BENTHIC_2017 \
    --output_path aesa_BENTHIC_2017_train.record --label_map_path  aesa_benthic_label_map.pbtxt --set train 
python create_tfrecord.py  
    --data_dir PATH_TO_TRAINING_DATA --collection aesa_BENTHIC_2017 \
    --output_path aesa_BENTHIC_2017_test.record --label_map_path  aesa_benthic_label_map.pbtxt --set test 
```    

## Download pretrained models for transfer learning
``` bash
mkdir -p models/
cd models
curl http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2017_11_08.tar.gz > faster_rcnn_resnet101_coco_11_08_2017.tar.gz
tar -xvf faster_rcnn_resnet101_coco_11_08_2017.tar.gz 

# optional if you have enough memory
curl http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_2017_11_08.tar.gz > faster_rcnn_inception_resnet_v2_atrous_coco_2017_11_08.tar.gz
tar -xvf faster_rcnn_inception_resnet_v2_atrous_coco_2017_11_08.tar.gz  
```
    
### Edit the pipeline.config file
Insert the correct paths for the training/test data in the train/test_input_reader and num_examples in the eval_config

### Train the model 
``` bash     
python tensorflow_models/research/object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=`pwd`/models/faster_rcnn_resnet101_coco960540resolution_smallanchor/pipeline.config \ 
    --train_dir=`pwd`/models/faster_rcnn_resnet101_coco960540resolution_smallanchor/checkpoints \ 
    --eval_dir=`pwd`/models/faster_rcnn_resnet101_coco960540resolution_smallanchor/eval
```
      
### Test the model 
``` bash
python tensorflow_models/research/object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=`pwd`/models/faster_rcnn_resnet101_coco960540resolution_smallanchor/pipeline.config \ 
    --checkpoint_dir=`pwd`/models/checkpoints/ \
    --eval_dir=PATH_TO_EVAL_DIR
```
 

## Annotation totals
For raw annoations
*  11748 examples in train set
*  3057 examples in test set

*Optimized* 
Using conf.OPTIMIZE_BOX
*  13906 examples in train set
*  3462 examples in test set

## Developer Notes
BUG in 
tensorflow_models/research/object_detection/inference/detection_inference.py
line 
 with tf.gfile.Open(inference_graph_path, 'r') as graph_def_file:
should be
 with tf.gfile.Open(inference_graph_path, 'rb') as graph_def_file:
for python3.x

# To evaluate model time, insert
 t = time.perf_counter()
 elapsed_time = time.perf_counter() - t
 print('Elapsed time {0}'.format(t))
in lines 84 and 91 of 
tensorflow_models/research/object_detection/inference/infer_detections.py

A placeholder for notes that might be useful for developers
* Pre processing options [https://github.com/tensorflow/models/blob/master/object_detection/protos/preprocessor.proto](https://github.com/tensorflow/models/blob/master/object_detection/protos/preprocessor.proto) 
* Install your own dataset [https://github.com/tensorflow/models/blob/master/object_detection/g3doc/using_your_own_dataset.md](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/using_your_own_dataset.md)
* Install TensorFlow Object Detection API [dhttps://github.com/tensorflow/models/blob/master/object_detection/g3doc/installation.m](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/installation.md) 
* Running in the cloud [https://cloud.google.com/blog/big-data/2017/09/performing-prediction-with-tensorflow-object-detection-models-on-google-cloud-machine-learning-engine](https://cloud.google.com/blog/big-data/2017/09/performing-prediction-with-tensorflow-object-detection-models-on-google-cloud-machine-learning-engine)
* Configuring option detection pipeline [https://github.com/tensorflow/models/blob/master/object_detection/g3doc/configuring_jobs.md](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/configuring_jobs.md)
* To see GPU usage on DevBox 
    $ watch nvidia-smi 
* To run train/eval on different GPUS, add to train/eval.py
  import os
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = "0" or "1" # to use first or second GPU
* How to see more boxes in Tensorboard [https://stackoverflow.com/questions/45452376/small-object-detection-with-faster-rcnn-in-tensorflow-models](https://stackoverflow.com/questions/45452376/small-object-detection-with-faster-rcnn-in-tensorflow-models)
* Good overview article on the different detection methods [https://medium.com/towards-data-science/deep-learning-for-object-detection-a-comprehensive-review-73930816d8d9](https://medium.com/towards-data-science/deep-learning-for-object-detection-a-comprehensive-review-73930816d8d9) 
