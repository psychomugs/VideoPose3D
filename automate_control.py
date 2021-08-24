import glob
import os

if __name__=="__main__":
    for user_id in ['B14pj','p5ebz']:
        for file in glob.glob(f'./inference/output_dir/*{user_id}*.mp4'):
            seq_name = file.split('/')[-1].replace('.mp4','')
            print(seq_name)
            os.system(f"python3 control/control.py --video {seq_name}")