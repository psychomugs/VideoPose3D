import os
import glob
import datetime

def transfer(video=None, is_dir=False):
    scp_cmd = 'scp '
    if is_dir: scp_cmd += '-r '
    os.system(scp_cmd+'../blossom/src/sequences/woody/vp/* dflocal:/private/home/msuguitan/projects/mvmt_ml/data/vp/')
    # os.system('scp /Users/msuguitan/Downloads/nick.webm $DEVFAIR/projects/videopose3d/inference/input_dir/nick_30fps.webm'.format(vp_webms[-1]))

if __name__=="__main__":
    transfer()
