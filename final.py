#!/usr/bin/python2
from copy import deepcopy
from time import sleep
import numpy as np
import rospy
import sys
from random import random as rand

import time
from os import getcwd
sys.path.append(getcwd() + "/../Core")

from arm_controller import ArmController
from astar import Astar
from loadmap import loadmap

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('usage: python final.py <color>')
        sys.exit()

    color = sys.argv[1]
    lynx = ArmController(color)

    sleep(2) # wait for setup

    map_struct = loadmap("./maps/final.txt")
    start = np.array([0, 0, 0, 0, 0, 0])

    lynx.set_pos(start)
    sleep(5)

    # interact with simulator, such as...

    # get state of your robot
    [q, qd] = lynx.get_state()
    print(q)
    print(qd)

    # get state of scoreable objects
    [name, pose, twist] = lynx.get_object_state()
    names = np.array(name)
    poses = np.array(pose)
    twists = np.array(twist)
    start = time.time()
    dynamic_indices = [i for i, item in enumerate(names) if "dynamic" in item]
    static_indices = [i for i in range(len(names)) if not i in dynamic_indices]
    print(dynamic_indices)
    print(static_indices)
    d_pose = poses[dynamic_indices]
    s_pose = poses[static_indices]
    d_twist = twists[dynamic_indices]
    s_twist = twists[static_indices]
    print(s_pose[0])
    print(time.time() - start)

    upper_lim = [1.4, 1.4, 1.7, 1.7, 1.5, -15]
    lower_lim = [-1.4, -1.2, -1.8, -1.9, -2.0, 30]

    down = np.array([0, 1.4, -1.3, 0, -np.pi/2, 30])
    lynx.set_pos(down)
    sleep(5)
    
    # down = np.array([1.4, 1.4, -1.25, 0, -np.pi/2, 30])
    # lynx.set_pos(down)
    # sleep(5)

    # # get state of your opponent's robot
    # [q, qd] = lynx.get_opponent_state()
    # print(q)
    # print(qd)

    lynx.stop()
