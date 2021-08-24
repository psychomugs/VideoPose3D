import os
import glob
import datetime
import sys
import shutil

def transfer(video, is_dir=False):
    print(f"Transferring {video}")
    # scp_cmd = 'sudo sshpass -p $uGu17a1r scp '
    scp_cmd = 'scp '
    # scp_cmd += '-i ~/.ssh/id_rsa.pub '
    if is_dir: scp_cmd += '-r '
    # os.system(scp_cmd+f'{video} $DEVFAIR/projects/videopose3d/inference/input_dir/')

    os.system(scp_cmd+f'{video} dflocal:/private/home/msuguitan/projects/videopose3d/inference/input_dir/')
    # os.system('scp /Users/msuguitan/Downloads/nick.webm $DEVFAIR/projects/videopose3d/inference/input_dir/nick_30fps.webm'.format(vp_webms[-1]))



def get_vp_webms():
    return sorted(glob.glob(os.path.expanduser('~')+'/Downloads/vp_*'))

def pull():
    # input("Press enter when inference is done to pull from remote ")
    os.system('rsync -vau --ignore-existing dflocal:/private/home/msuguitan/projects/videopose3d/inference/old_output_dir/* ./inference/output_dir/')


if __name__=="__main__":
    webm_file = '../blossom-app/public/webms.txt'
    # pull()
    # exit()
    if len(sys.argv)>1:
        transfer(sys.argv[1])
    else:
        while 1:
            cmd = input("Enter (t)ransfer most recent webm / (p)ull / or manually enter webm name: ")
            if cmd=='t':
                newest_webm = get_vp_webms()[-1]
                transfer(newest_webm)
                shutil.copy2(newest_webm, '../blossom-app/public/vid.webm')

                seq_name = 'vp_'+'_'.join(newest_webm.split('.')[0].split('_')[-3:])

                with open(webm_file,'a') as f: f.write(f"\n{seq_name}")
            elif cmd=='p':
                pull()
                os.system('python3 control/control.py --video newest')
            else:
                transfer(cmd)


    # for v in glob.glob(os.path.expanduser('~')+'/projects/videos/lia_kim/*'):
    #     transfer(v)
    # transfer(os.path.expanduser('~')+'/projects/videos/lia_kim/*', is_dir=True)

    # transfer(os.path.expanduser('~')+'/projects/videos/vrajdance_trimmed.mp4')

    # pull()