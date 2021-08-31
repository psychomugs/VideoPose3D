# -*- coding: utf-8 -*-


import os
import ffmpeg
from argparse import Namespace
from inference import infer_video_d2
import sys
import subprocess
import glob
sys.path.insert(0,'./data/')
from data import prepare_data_2d_custom

import time

def get_length(input_video):
    print(input_video)
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', input_video], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return float(result.stdout)

if __name__=="__main__":
    start_time = time.time()

    target_framerate = 30

    os.chdir('./inference/')
    input_vids = os.listdir('./input_dir')
    vid_lengths = []
    print(input_vids)
    for input_vid in input_vids:
        # skip already converted videos
        if f'_{target_framerate}fps' in input_vid: continue
        stream = ffmpeg.input(f'./input_dir/{input_vid}').filter('fps', fps=target_framerate, round='up')
        output_vid = input_vid
        if 'vp' in input_vid[:2]:
            [_,date,user_id,arms,scenario] = input_vid.split('_')
            output_vid
        output_name = f'./input_dir/{input_vid.replace(".webm",f"_{target_framerate}fps.mp4")}'
        stream = ffmpeg.output(stream, output_name)
        ffmpeg.run(stream)
        vid_lengths.append(get_length(output_name))
        # remove input video
        os.remove(f'./input_dir/{input_vid}')

    infer_video_d2.setup_logger()
    args = Namespace(**{
        'cfg':'COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml',
        'output_dir':'output_dir',
        'image_ext':'mp4',
        'im_or_folder':'input_dir'
    })
    print(args)
    infer_video_d2.main(args)

    os.chdir('../data/')
    args = Namespace(**{
        'input': '../inference/output_dir',
        'output': 'myvideos'
    })
    prepare_data_2d_custom.main(args)

    os.chdir('../')

    for input_vid in os.listdir('./inference/input_dir'):
        vid_name = input_vid.split('.')[0]
        exec_cmd = f"python3 run.py -d custom -k myvideos -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-subject {input_vid} --viz-action custom --viz-camera 0 --viz-video inference/input_dir/{input_vid} --viz-output inference/output_dir/{vid_name}.mp4 --viz-export inference/output_dir/{vid_name}.npy --viz-size 6"
        os.system(exec_cmd)

    # move input,output files to "old" dirs
    for src in ['input','output']:
        src_files = glob.glob(f'./inference/{src}_dir/*')
        tgt_files = [s.replace(f'/{src}',f'/old_{src}') for s in src_files]
        [os.rename(s,t) for s,t in zip(src_files,tgt_files)]


    delta_time = int(time.time()-start_time)
    print(f"Took {delta_time}s for {len(input_vids)} video(s) totaling {sum(vid_lengths)}s: {delta_time/sum(vid_lengths)}x")