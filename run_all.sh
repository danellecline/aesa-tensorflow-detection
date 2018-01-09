#!/bin/bash
nohup ./run.sh faster_rcnn_resnet101_coco960540resolution_smallanchor train 0 > faster_rcnn_resnet101_coco960540resolution_smallanchor.out 2>&1 &
#nohup ./run.sh faster_rcnn_resnet101_coco960540resolution_smallanchor_meanadjusted train 1 > faster_rcnn_resnet101_coco960540resolution_smallanchor_meanadjusted.out 2>&1 &
#nohup ./run.sh faster_rcnn_resnet101_coco960540resolution_smallanchor_contrastaug train 2 > faster_rcnn_resnet101_coco960540resolution_smallanchor_contrastaug.out 2>&1 &
