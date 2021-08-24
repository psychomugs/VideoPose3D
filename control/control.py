import requests
import time
import sys
import os
import glob

sys.path.append('../blossom/')
from src import server, sequence

import socketio
sio = socketio.Client()
from scipy.spatial.transform import Rotation as R
import numpy as np
from utils import Keypoints3D as K
from utils import Axis3D as A

# framerate = 30.
# framerate = 15.
target_framerate = 10.
# target_framerate = 30.
import cv2
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import PIL
default_height = 80
last_height = default_height
# embodiments = ['NoArms','OneArm','TwoArms']


import argparse
def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--video','-v',default='',
        help='The name of the video the robot will mimic. Accepts either video.mp4, dir/video.mp4, etc')
    parser.add_argument('--height', '-height', default=False,
        help='Whether to implement height control or not.',
        action='store_true')
    parser.add_argument('--robot','-robot', default=True,
        help='Whether to control the robot or not.',
        action='store_true')
    return parser.parse_args(args)


def load_numpy(np_path):
    return np.load(np_path, allow_pickle=True)


def get_ctrl_addr(path='../blossom-app/public/ctrl_addr.txt'):
    with open(path) as f:
        return f.read().replace('\n','')


def get_heading(x,y,z):
    cX, cY, cZ = np.cos(x), np.cos(y), np.cos(z)
    sX, sY, sZ = np.sin(x), np.sin(y), np.sin(z)
    Vx = -cZ*sY - sZ*sX*cY
    Vy = -sZ*sY + cZ*sX*cY
    compass_heading = np.arctan2(Vy,Vx)
    if Vy<0:
        compass_heading += np.pi 
    elif Vx<0: 
        compass_heading += 2*np.pi
    return compass_heading


def send_data(ctrl_addr, rot_dict, arm_pos, height):
    data_packet = {
        'x': rot_dict['x'],
        'y': rot_dict['y'],
        'z': -rot_dict['z'],
        'ax': 0,
        'ay': 0,
        'az': 0,
        'h': int(height),
        # // h: parseInt(heightPos),
        'q': 0,
        'ears': 50,
        'left_arm': arm_pos['left'],
        'right_arm': arm_pos['right'],
        'landscape': False,
        'mirror': False,
        'heightCtrl': False,
        'yaw': 0,
        'time': time.time(),
        'portrait': False,
      };
    requests.post(ctrl_addr+'position', json=data_packet)
    motor_pos = server.imu_to_motor_pos(data_packet, sensitivity=3.0)
    motor_pos.update({k:(v/50. + 3) for k,v in motor_pos.items()})
    print(motor_pos)
    return motor_pos


def get_rot(x,y,z,rot_order='ZXY'):
    rot = R.align_vectors(
            # [[1,0,0],[0,1,0],[0,0,1]],
            # [[0,0,1],[0,1,0],[-1,0,0]],
            [[x,y,z]],

            [[0,0,1]],
            )[0].as_euler(rot_order)
    rot_dict = {c:r for c,r in zip(rot_order.lower(),rot)}
    # [x,y,z] = R.align_vectors([[x,y,z]],[[1,0,0]])[0].as_euler(rot_order)
    # rot_dict['z'] = get_heading(x,y,z)
    # z -= np.pi
    print([rot_dict[c] for c in 'xyz'])
    if rot_dict['z']<0 : rot_dict['z'] += 2*np.pi
    if rot_dict['z']>2*np.pi: rot_dict['z'] -= 2*np.pi
    print([rot_dict[c] for c in 'xyz'])
    return rot_dict

def get_rot_dict(x,y,z,rot_order='ZXY'):
    rot = R.align_vectors(
        [x,y,z],
        [[1,0,0],[0,1,0],[0,0,1]],
    )[0].as_euler(rot_order)
    rot_dict = {c:r for c,r in zip(rot_order.lower(),rot)}
    # [x,y,z] = R.align_vectors([[x,y,z]],[[1,0,0]])[0].as_euler(rot_order)
    # rot_dict['z'] = get_heading(x,y,z)
    # z -= np.pi
    # print([rot_dict[c] for c in 'xyz'])
    # if rot_dict['z']<0 : rot_dict['z'] += 2*np.pi
    # if rot_dict['z']>2*np.pi: rot_dict['z'] -= 2*np.pi
    # print([rot_dict[c] for c in 'xyz'])
    return rot_dict


# def rot_dict2frame(rot_dict):
#     # return [[[0,0,0],rot] for rot in list(rot_dict.values())]



def live_input(ctrl_addr):
    while True:
        pos_input = input('x/y/z pos: ')
        [x,y,z] = [int(pos) for pos in pos_input.split(',')]
        # [x,y,z] = [0]*3
        z *= -1
        rot_dict = {c:(r*np.pi/180) for c,r in zip('xyz',[x,y,z])}
        # rot_order = 'zxy'
        # rot_order = 'ZXY'
        # rot_dict = get_rot(x,y,z,rot_order)
        arm_pos = 0
        send_data(ctrl_addr, rot_dict, arm_pos)


def np_input(ctrl_addr, input_file, draw=True, height_control=False, arm_mode=2, robot_control=True, record=False):


    #vp_2021-08-03T23-40_31-352Z_i772O_NoArms_Happy1
    # parse the file name 
    if 'vp' in input_file[:2]:
        # [user_id,embodiment,scenario] = input_file.split('_')[2:5]
        [scenario,embodiment,user_id] = input_file.split('_')[2:5]
        save_fn = f'vp/{scenario.lower()}_{embodiment}_{user_id}'
        arm_mode = int(embodiment[-1])
    else:
        save_fn = input_file
    print(f"Using {arm_mode} arms")


    np_file = './inference/output_dir/{}.npy'.format(input_file)
    frames = load_numpy(np_file)
    frames *= -1

    # swap y and z axes
    frames[...,[1,2]] = frames[...,[2,1]]

    # print(frames.shape)
    vid = cv2.VideoCapture(np_file.replace('npy','mp4'))
    vidsize = (
        int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    framerate = vid.get(cv2.CAP_PROP_FPS)
    frame_skip = int(framerate/target_framerate)
    # manually setting frame_skip to 4 @30fps but target @10fps works best
    # frame_skip = 4
    print(f"Actual / Target framerates: {framerate} / {target_framerate}, skipping {frame_skip} frame(s)")
    vid_writer = cv2.VideoWriter(
        './control/control_vids/{}.mp4'.format(input_file),
        fourcc,
        framerate/frame_skip,
        (vidsize[0],vidsize[1]*2),
        True)
    # fig,ax = plt.subplots(figsize=vidsize)
    # print(vidsize)
    fig = plt.figure(figsize=tuple([v/100 for v in vidsize]))
    ax = plt.axes(projection='3d')
    # os.system('open {}'.format(np_file.replace('npy','mp4')))

    l_shoulder_0,r_shoulder_0 = frames[0][K.L_SHOULDER.value], frames[0][K.R_SHOULDER.value]
    root = np.average([l_shoulder_0,r_shoulder_0],axis=0)
    print(f"Root position: {root}")
    root[2] = 0
    # root = 
    # print(shoulder_0)
    def _get_rot(frame):
        # Get the rotation matrix for the upper body

        # root = frame[K.SPINE.value]
        # root = frame[K.R_HIP.value]+frame[K.L_HIP.value]

        # z_rot = R.from_rotvec([0,0,np.pi/4])
        # get shoulder locations = vectors because centered around <0,0,0>
        # root = frame[K.SPINE.value]
        # root = [0,0,np.average([l_shoulder_0[2],r_shoulder_0[2]])]
        # root = shoulder
        # print(root)
        # print(root)
        # root = frame[K.]
        l_shoulder, r_shoulder = frame[K.L_SHOULDER.value]-root, frame[K.R_SHOULDER.value]-root
        # l_shoulder, r_shoulder = -frame[K.L_EAR.value], -frame[K.R_EAR.value]
        # reverse Z and Y axes
        # l_shoulder[[1,2]], r_shoulder[[1,2]] = l_shoulder[[2,1]], r_shoulder[[2,1]]
        l_norm, r_norm = l_shoulder, r_shoulder
        # [l_norm,r_norm] = z_rot.apply([l_norm,r_norm])
        # l_norm, r_norm = l_shoulder/np.linalg.norm(l_shoulder), r_shoulder/np.linalg.norm(r_shoulder)
        # calculate transformed frame
        z = (l_norm+r_norm)
        z /= np.linalg.norm(z)
        y = np.cross(l_norm,r_norm)
        y /= np.linalg.norm(y)
        # print(z,y)
        x = np.cross(y,z)
        x /= np.linalg.norm(x)



        # EXPERIMENTAL - use head instead of shoulders
        if True:
            # z = frame[K.HEAD.value]-frame[K.SPINE.value]
            z = frame[K.HEAD.value]-frame[K.THORAX.value]
            # z = frame[K.NOSE.value]-frame[K.HEAD.value]
            z /= np.linalg.norm(z)
            x = frame[K.R_SHOULDER.value]-frame[K.L_SHOULDER.value]
            x /= np.linalg.norm(x)
            y = np.cross(z,x)


        # print(x,y,z)        else:
        return get_rot_dict(x,y,z,rot_order='ZXY'), [l_shoulder, r_shoulder, x, y, z]

    norm = np.linalg.norm
    _get_vec = lambda x0,x1 : frame[x1.value]-frame[x0.value]
    _get_angle = lambda a,b,c : np.arccos(
        (norm(a)**2 + norm(b)**2 - norm(c)**2) / (2*norm(a)*norm(b)))*180/np.pi

    arm_rest = 10
    [l_arm_rest, r_arm_rest] = list(np.random.uniform(low=-arm_rest,high=arm_rest,size=2))
    [l_arm_amp, r_arm_amp] = list(np.random.uniform(low=-arm_rest,high=arm_rest,size=2))
    [l_arm_start,r_arm_start] = list(np.random.uniform(low=-arm_rest*10,high=arm_rest*10,size=2))

    print(l_arm_rest, l_arm_amp, l_arm_start)
    arm_offset = 100
    def _get_arm_pos(frame, f, angle_multiplier=1.5):
        # Get the arm positions

        a,b,c = '{}_SHOULDER','{}_ELBOW','WRIST'
        # l_upper_arm = _get_vec(K.L_SHOULDER, K.L_ELBOW)
        # r_upper_arm = _get_vec(K.R_SHOULDER, K.R_ELBOW)
        # l_lower_arm = _get_vec(K.L_ELBOW, K.L_WRIST)
        # r_lower_arm = _get_vec(K.R_ELBOW, K.R_WRIST)
        # l_angle = -150+_get_angle(l_upper_arm,l_lower_arm,l_arm)
        # r_angle = 150-_get_angle(r_upper_arm,r_lower_arm,r_arm)

        l_upper_arm = _get_vec(K.L_HIP, K.L_SHOULDER)
        r_upper_arm = _get_vec(K.R_HIP, K.R_SHOULDER)
        l_lower_arm = _get_vec(K.L_ELBOW, K.L_WRIST)
        r_lower_arm = _get_vec(K.R_ELBOW, K.R_WRIST)

        l_arm = l_upper_arm+l_lower_arm
        r_arm = r_upper_arm+r_lower_arm
        angle_mul, angle_pow = 1.5,1
        _amplify = lambda angle : angle_mul*150*((angle/150)**angle_pow)    
        l_angle = -_amplify(_get_angle(l_upper_arm,l_lower_arm,l_arm))
        r_angle = _amplify(_get_angle(r_upper_arm,r_lower_arm,r_arm))
        # print(r_angle,l_angle)

        # if less than both arms, drop left arm
        if arm_mode<2:
            l_angle = l_arm_rest + l_arm_amp*np.sin((f+l_arm_start)*np.pi/180)
        # if no arms, also drop right arm
        if arm_mode<1:
            r_angle = r_arm_rest + r_arm_amp*np.sin((f+r_arm_start)*np.pi/180)

        # account for offsets
        l_angle += 100
        r_angle -= 100

        return {'left':l_angle, 'right':r_angle}, [l_lower_arm, r_lower_arm]

    def _get_height(frame):
        global last_height
        l_z = [frame[k.value][-1] for k in [K.L_FOOT,K.L_KNEE,K.L_HIP]]
        r_z = [frame[k.value][-1] for k in [K.R_FOOT,K.R_KNEE,K.R_HIP]]

        l_lower_leg = _get_vec(K.L_FOOT,K.L_KNEE)
        l_upper_leg = _get_vec(K.L_KNEE,K.L_HIP)
        r_lower_leg = _get_vec(K.R_FOOT,K.R_KNEE)
        r_upper_leg = _get_vec(K.R_KNEE,K.R_HIP)
        l_height_ratio = norm(l_lower_leg+l_upper_leg)/(norm(l_lower_leg)+norm(l_upper_leg))
        r_height_ratio = norm(r_lower_leg+r_upper_leg)/(norm(r_lower_leg)+norm(r_upper_leg))
        height = np.average([l_height_ratio,r_height_ratio])
        # check if legs are messed up - if so set to default height
        _leg_violation = lambda leg : leg[0]>leg[1] or leg[0]>leg[2] or leg[1]>leg[2] or height<0.5
        if _leg_violation(l_z) or _leg_violation(r_z):
            height = last_height
        else:
            height = np.interp(height**2,
                [0.6**2,1.0],
                [20,80],
                )
        last_height = height
        # print(height)
        return height

    # start recording
    if record: requests.post(ctrl_addr+'record/start')

    # for frame in frames:
    ret,img = vid.read()
    # frame_skip *=2
    # frame_skip = int(frame_skip*1.4)
    start_frame = 0
    end_frame = len(frames)
    end_frame -= int(framerate)

    # for f in range(0,len(frames),frame_skip):

    # cut out last second to account for me going to click the mouse
    # for f in range(0,len(frames)-1*framerate,frame_skip):
    motor_pos_list = []
    for f in range(start_frame, end_frame, frame_skip):
        t = time.time()

        frame = frames[f]
        # rot_dict = {c:0 for c in 'xyz'}
        arm_pos, arm_vectors = _get_arm_pos(frame, f)
        rot_dict, vectors = _get_rot(frame)
        height = _get_height(frame) if height_control else default_height

        # rot_frame = rot_dict2frame(rot_dict)
        if draw:
            ax.clear()
            ax.view_init(30,90)
            vec_colors = ['r','b','g','c','m']
            for rot,color in zip(vectors,vec_colors):
                rot_vec = [[0,r] for r in rot]
                ax.plot(rot_vec[0],rot_vec[1],rot_vec[2],color=color)
            for shoulder,arm in zip(vectors[:2],arm_vectors):
                arm_vec = [[s,a] for s,a in zip(shoulder,arm+shoulder)]
                ax.plot(arm_vec[0],arm_vec[1],arm_vec[2])
            for set_lim in [ax.set_xlim, ax.set_ylim, ax.set_zlim]:
                set_lim([-1,1])
            fig.canvas.draw()
            fig_np = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            fig_np = fig_np.reshape(fig.canvas.get_width_height()[::-1]+(3,))
            concat_img = cv2.vconcat([img,fig_np])
            vid_writer.write(concat_img)

            cv2.imshow('Frame',concat_img)
            if cv2.waitKey(25) & 0xFF==ord('q'):
                break
        motor_pos_list.append(send_data(ctrl_addr, rot_dict, arm_pos, height))

        # time.sleep(1./target_framerate)
        
        # while (time.time()-t<(1./target_framerate)):
        #     pass

        # input('Press Enter to step to next frame ')
        for _ in range(frame_skip):
            ret,img = vid.read()
            if not ret: break

    frame_frequency = int(1000/target_framerate)

    millis = range(0,len(motor_pos_list)*frame_frequency, frame_frequency)
    # print(save_fn)
    print(sequence.Sequence.from_dict(
        millis, motor_pos_list, save_fn
        ).to_file(robot_dir='../blossom/src/sequences/woody/'))

    if record: requests.post(ctrl_addr+'record/stop/{}'.format(save_fn))
    requests.get(f"{ctrl_addr}/r")

    vid.release()
    vid_writer.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    ctrl_addr = 'http://localhost:4000/'
    # live_input(ctrl_addr)
    start_time = time.time()
    args = parse_args(sys.argv[1:])
    print(args.video)

    input_vid = args.video.split('/')[-1].split('.')[0]
    if len(input_vid)==0: 
        print(f'Invalid video {input_vid} specified.')
    elif input_vid=='newest':
        input_vid = sorted(glob.glob('inference/output_dir/vp*.npy'))[-1]
        input_vid = input_vid.split('/')[-1].replace('.npy','')
    else:
        print(f'Running video {input_vid}.')
    np_input(ctrl_addr, input_vid, height_control=args.height, robot_control=args.robot)
    print("took {} seconds".format(time.time()-start_time))
