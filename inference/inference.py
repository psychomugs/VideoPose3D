import os
import ffmpeg
from argparse import Namespace
from inference import infer_video_d2


if __name__=="__main__":
    os.chdir('./inference')
    input_vids = os.listdir('./input_dir')
    print(input_vids)
    for input_vid in input_vids:
        # skip already converted videos
        if '_30fps' in input_vid: continue
        stream = ffmpeg.input(f'./input_dir/{input_vid}')
        stream = ffmpeg.output(stream, f'./input_dir/{input_vid.replace(".mp4","_30fps.mp4")}')
        ffmpeg.run(stream)
        print(f'Removing {input_vid}')
        os.remove(f'./input_dir/{input_vid}')
    # ffmpeg.run(ffmpeg.output())

    infer_video_d2.setup_logger()
    args = Namespace(**{
        'cfg':'COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml',
        'output_dir':'output_dir',
        'image_ext':'mp4',
        'im_or_folder':'input_dir'
    })
    print(args)
    infer_video_d2.main(args)


 
# Namespace(cfg='COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml', im_or_folder='input_dir', image_ext='mp4', output_dir='output_dir')