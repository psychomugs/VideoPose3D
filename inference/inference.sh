#!/bin/sh
python3 infer_video_d2.py \
    --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml \
    --output-dir monologue_output \
    --image-ext mp4 \
    input_dir

cd ../data
python3 prepare_data_2d_custom.py -i ../inference/monologue_output -o myvideos

cd ..
for FILE in inference/input_dir/* ;
    do f="$(echo $FILE | cut -d'/' -f3)";
    fname="$(echo $f | cut -d'.' -f1)";
    echo $fname;
    python3 run.py -d custom -k myvideos -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-subject $f --viz-action custom --viz-camera 0 --viz-video $FILE --viz-output inference/monologue_output/$fname.mp4 --viz-size 6;
    python3 run.py -d custom -k myvideos -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-subject $f --viz-action custom --viz-camera 0 --viz-video $FILE --viz-export inference/monologue_output/$fname.npy --viz-size 6;
done