#!/bin/sh
python3 infer_video_d2.py \
    --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml \
    --output-dir output_dir \
    --image-ext mp4 \
    input_dir