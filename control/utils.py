from enum import Enum

class Axis3D(Enum):
    X = 0
    Z = 1
    Y = 2

class Keypoints3D(Enum):
    HIP = 0
    R_HIP = 1
    R_KNEE = 2
    R_FOOT = 3
    L_HIP = 4
    L_KNEE = 5
    L_FOOT = 6
    SPINE = 7
    THORAX = 8
    NOSE = 9
    HEAD = 10
    L_SHOULDER = 11
    L_ELBOW = 12
    L_WRIST = 13
    R_SHOULDER = 14
    R_ELBOW = 15
    R_WRIST = 16