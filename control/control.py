import requests
import time
import sys
import os

import socketio
sio = socketio.Client()
from scipy.spatial.transform import Rotation as R
import numpy as np
from utils import Keypoints3D as K
from utils import Axis3D as A

framerate = 30.
import cv2


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


def send_data(ctrl_addr, rot_dict, arm_pos):
    data_packet = {
        'x': rot_dict['x'],
        'y': rot_dict['y'],
        'z': rot_dict['z'],
        'ax': 0,
        'ay': 0,
        'az': 0,
        'h': 70,
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
    requests.post(ctrl_addr, json=data_packet)


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

def get_rot_frame(x,y,z,rot_order='ZXY'):
    rot = R.align_vectors(
        [x,y,z],
        [[1,0,0],[0,1,0],[0,0,1]],
    )[0].as_euler(rot_order)
    rot_dict = {c:r for c,r in zip(rot_order.lower(),rot)}
    # [x,y,z] = R.align_vectors([[x,y,z]],[[1,0,0]])[0].as_euler(rot_order)
    # rot_dict['z'] = get_heading(x,y,z)
    # z -= np.pi
    # print([rot_dict[c] for c in 'xyz'])
    if rot_dict['z']<0 : rot_dict['z'] += 2*np.pi
    if rot_dict['z']>2*np.pi: rot_dict['z'] -= 2*np.pi
    print([rot_dict[c] for c in 'xyz'])
    return rot_dict


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


def np_input(ctrl_addr, input_file):
    np_file = './inference/output_dir/{}.npy'.format(input_file)
    frames = load_numpy(np_file)
    # print(frames.shape)
    vid = cv2.VideoCapture(np_file.replace('npy','mp4'))
    # os.system('open {}'.format(np_file.replace('npy','mp4')))

    def _get_rot(frame):
        # get shoulder locations = vectors because centered around <0,0,0>
        l_shoulder, r_shoulder = -frame[K.L_SHOULDER.value], -frame[K.R_SHOULDER.value]
        # l_shoulder, r_shoulder = -frame[K.L_EAR.value], -frame[K.R_EAR.value]
        l_shoulder[[1,2]], r_shoulder[[1,2]] = l_shoulder[[2,1]], r_shoulder[[2,1]]
        l_norm, r_norm = l_shoulder/np.linalg.norm(l_shoulder), r_shoulder/np.linalg.norm(r_shoulder)
        # calculate transformed frame
        z = (l_norm+r_norm)
        z /= np.linalg.norm(z)
        y = np.cross(l_shoulder,r_shoulder)
        y /= np.linalg.norm(y)
        # print(z,y)
        x = np.cross(y,z)
        x /= np.linalg.norm(x)
        print(x,y,z)

        return get_rot_frame(x,y,z,rot_order='ZXY')

    def _get_arm_pos(frame):
        return {'left':frame[K.L_WRIST.value][A.Z.value]*300,
            'right':-frame[K.R_WRIST.value][A.Z.value]*300}

    for frame in frames:
        # input('Press Enter to step to next frame ')
        ret,img = vid.read()
        # rot_dict = {c:0 for c in 'xyz'}
        arm_pos = _get_arm_pos(frame)
        rot_dict= _get_rot(frame)
        send_data(ctrl_addr, rot_dict, arm_pos)
        cv2.imshow('Frame',img)
        if cv2.waitKey(25) & 0xFF==ord('q'):
            break
        # time.sleep(1./framerate)

    vid.release()
    cv2.destroyAllWindows()



if __name__=="__main__":
    ctrl_addr = 'http://localhost:4000/position'
    # live_input(ctrl_addr)
    np_input(ctrl_addr, sys.argv[1])







