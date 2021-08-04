import os
import glob
import datetime

def transfer(video, is_dir=False):
    scp_cmd = 'scp '
    if is_dir: scp_cmd += '-r '
    os.system(scp_cmd+'{} $DEVFAIR/projects/videopose3d/inference/input_dir/'.format(video))
    # os.system('scp /Users/msuguitan/Downloads/nick.webm $DEVFAIR/projects/videopose3d/inference/input_dir/nick_30fps.webm'.format(vp_webms[-1]))



def get_vp_webms():
    return sorted(glob.glob(os.path.expanduser('~')+'/Downloads/vp_*'))

def pull():
    input("Press enter when inference is done to pull from remote ")
    os.system('rsync -vau --ignore-existing $DEVFAIR/projects/videopose3d/inference/old_output_dir/* ./inference/output_dir/')


if __name__=="__main__":
    transfer(get_vp_webms()[-1])

    # for v in glob.glob(os.path.expanduser('~')+'/projects/videos/lia_kim/*'):
    #     transfer(v)
    # transfer(os.path.expanduser('~')+'/projects/videos/lia_kim/*', is_dir=True)

    # transfer(os.path.expanduser('~')+'/projects/videos/vrajdance_trimmed.mp4')

    pull()