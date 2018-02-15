# aesa-tensorflow-detection 
 
Code for testing Tensorflow detection API on AESA data

Mosaic images are broken into 500x00 tiles for training/testing.

[![ Image link ](/img/summary.png)]

## Prerequisites
 
- Python version  3.5
- [Protobuf](https://developers.google.com/protocol-buffers/
- Packages: python-tk e.g. apt-get install python3.5-tk,  
- Access to the processed mosaic data and annotations that accompany those. Send an email request if interested

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
    --data_dir PATH_TO_TRAINING_DATA --collection M53545556_500x500 \
    --output_path M53545556_500x500_train.record --label_map_path  aesa_benthic_label_map.pbtxt --set train 
python create_tfrecord.py  
    --data_dir PATH_TO_TRAINING_DATA --collection M53545556_500x500 \
    --output_path M53545556_500x500_test.record --label_map_path  aesa_benthic_label_map.pbtxt --set test 
```    

## Download pretrained models for transfer learning
``` bash
mkdir -p models/
cd models
curl -o faster_rcnn_resnet101_coco_11_08_2017.tar.gz http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2017_11_08.tar.gz 
curl -o rfcn_resnet101_coco_2018_01_28.tar.gz http://download.tensorflow.org/models/object_detection/rfcn_resnet101_coco_2018_01_28.tar.gz
curl -o ssd_inception_v2_coco_2017_11_17.tar.gz http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz
tar -xvf ssd_inception_v2_coco_2017_11_17.tar.gz
tar -xvf rfcn_resnet101_coco_2018_01_28.tar.gz.tar.gz 
tar -xvf faster_rcnn_resnet101_coco_11_08_2017.tar.gz 
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

TRAIN
----
* POLYCHAETA 313
* CRINOIDEA 328
* ASTEROIDEA 46
* CNIDARIA 9504
* ARTHROPODA 76
* OPHIUROIDEA 4748
* TUNICATA 557
* UNKNOWN 235
* PORIFERA 451
* HOLOTHUROIDEA 598
* ECHIURA 128
Done. Found 16984 examples in train set

TEST
----
* CNIDARIA 9539
* ARTHROPODA 64
* PORIFERA 424
* CRINOIDEA 347
* HOLOTHUROIDEA 623
* ASTEROIDEA 51
* OPHIUROIDEA 4766
* TUNICATA 573
* ECHIURA 138
* UNKNOWN 231
* POLYCHAETA 288
Done. Found 17044 examples in test set
 

## Developer Notes
BUG in 
tensorflow_models/research/object_detection/inference/detection_inference.py
line 
 with tf.gfile.Open(inference_graph_path, 'r') as graph_def_file:
should be
 with tf.gfile.Open(inference_graph_path, 'rb') as graph_def_file:
for python3.x

# To evaluate model time, insert
``` bash
import csv
import numpy as np
ofile  = open('{0}_gputime.csv'.format(FLAGS.output_tfrecord_path), 'wt')
writer = csv.writer(ofile)
writer.writerow(['GPU Time'])
times = []
t = time.process_time()
elapsed_time = time.process_time() - t
if counter > 0 :
      times.append(elapsed_time)
      m = np.mean(np.array(times))
      print('Elapsed time {0} mean {1}'.format(elapsed_time, m))

m = np.mean(times)
    writer.writerow(['{0}'.format(int(m*1000))])

 t = time.perf_counter()
 elapsed_time = time.perf_counter() - t
 print('Elapsed time {0}'.format(t))
starting at lines 76 of
tensorflow_models/research/object_detection/inference/infer_detections.py
```

# Developer notes

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
