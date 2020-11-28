#!/usr/bin/python2
from copy import deepcopy
from time import sleep
import numpy as np
import rospy
import sys
import utilities as util
from random import random as rand

from sys import path
from os import getcwd

path.append(getcwd() + "/../Core")

from arm_controller import ArmController
from astar import Astar
from loadmap import loadmap
from rrt import rrt

if __name__ == '__main__':
    # Update map location with the location of the target map
    map_struct = loadmap("./maps/map2.txt")
    start = np.array([0, 0, 0, 0, 0, 0])
    goal = np.array([-1.3, 0, 0.25, 0, 0.5, 0])

    # Run Astar code
    # path = Astar(deepcopy(map_struct), deepcopy(start), deepcopy(goal))

    # IMPORTANT HYPERPARAMETERS FOR RRT*FN ****************************
    max_nodes = 5000
    max_iter = 3000

    # or run rrt code
    path, cost = rrt(deepcopy(map_struct), deepcopy(start), deepcopy(goal), max_nodes, max_iter,
                     stepsize=0.1, neighbour_radius=0.15, bias_ratio=5, bias_radius=0.075,
                     optimize=True)

    # start ROS
    lynx = ArmController()
    sleep(1)  # wait for setup
    collision = False

    # Take out
    lynx.set_pos(start)
    sleep(5)

    # iterate over target waypoints
    for q in path:
        print("Goal:")
        print(q)

        lynx.set_pos(q)
        reached_target = False

        proximity_radius = 0.05
        while not reached_target:
            # Check if robot is collided then wait
            collision = collision or lynx.is_collided()
            if collision:
                break
            pos, vel = lynx.get_state()
            if util.reachedTarget(pos, q, proximity_radius):
                reached_target = True
            sleep(0.05)

        print("Current Configuration:")
        pos, vel = lynx.get_state()
        print(pos)

    if collision:
        print("Robot collided during move")
    else:
        print("No collision detected")

    lynx.stop()
