import numpy as np
import utilities as util
from utilities import getNearest, getNeighbours
from calculateFK import CalculateFK
from node import Node
from rrt_fn import choose_and_rewire_fn
import matplotlib.pyplot as plt
import time


def rrt_double(obstacles, start, goal, max_nodes, max_iter,
               stepsize=0.15, neighbour_radius=0.175, nodes={}):
    """
    Implement RRT algorithm in this file.
    :param obstacles:         the map struct
    :param start:       start pose of the robot (1x6).
    :param goal:        goal pose of the robot (1x6).
    :param max_nodes
    :param max_iter
    :param stepsize
    :param neighbour_radius
    :param nodes
    :return:            returns an mx6 matrix, where each row consists of the configuration of the Lynx at a point on
                        the path. The first row is start and the last row is goal. If no path is found, PATH is a 0x6
                        matrix..
    """

    root = Node(start, 0)
    nodes.update({np.array2string(root.angles, precision=5): root})
    end = Node(goal, 0)
    nodes_end = {}
    nodes_end.update({np.array2string(end.angles, precision=5): end})
    num_nodes = 1
    num_iter = 0
    first_path_found = False

    for i in range(max_iter):
        steer_point, steer_point_wk, nearest_point, nearest_point_wk = \
            get_sample_set(goal, root, stepsize, nodes)
        steer_point_end, steer_point_wk_end, nearest_point_end, nearest_point_wk_end = \
            get_sample_set(goal, end, stepsize, nodes_end)

        if num_nodes < max_nodes:
            # Root tree **********************************************************
            path_found, node, neighbour = \
                choose_and_rewire(root, end, steer_point, steer_point_wk, nearest_point,
                                  nearest_point_wk, obstacles, neighbour_radius, stepsize,
                                  nodes, nodes_end)
        else:
            # Root tree **********************************************************
            path_found, node, neighbour = \
                choose_and_rewire_fn(root, end, steer_point, steer_point_wk, nearest_point,
                                     nearest_point_wk, obstacles, neighbour_radius, stepsize,
                                     nodes, nodes_end)
        if path_found:
            first_path_found = True
            connectTrees(node, neighbour)
            print('path found!')
            # break and go to single rrt
            break


        if num_nodes < max_nodes:
            # Reverse tree **********************************************************
            reverse_path_found, node, neighbour = \
                choose_and_rewire(end, root, steer_point_end, steer_point_wk_end, nearest_point_end,
                                  nearest_point_wk_end, obstacles, neighbour_radius, stepsize,
                                  nodes_end, nodes)
        else:
            # Reverse tree **********************************************************
            reverse_path_found, node, neighbour = \
                choose_and_rewire_fn(end, root, steer_point_end, steer_point_wk_end, nearest_point_end,
                                     nearest_point_wk_end, obstacles, neighbour_radius, stepsize,
                                     nodes_end, nodes)
        if reverse_path_found:
            first_path_found = True
            connectTrees(neighbour, node)
            print('reverse path found!')
            # break and go to single rrt
            break

        if num_nodes < max_nodes:
            num_nodes += 2
        print("Iter: %d" % num_iter)
        num_iter += 1

    return first_path_found, root, end, num_iter, num_nodes


def get_sample_set(goal, root, stepsize, nodes):
    fk = CalculateFK()
    upper_lim = [1.4, 1.4, 1.7, 1.7, 1.5]
    lower_lim = [-1.4, -1.2, -1.8, -1.9, -2.0]
    random_point = util.sampleRandom(lower_lim, upper_lim, goal[4], goal[5])
    # nearest_point = root.getNearest(random_point)
    nearest_point = getNearest(random_point, nodes)
    nearest_point_wk = fk.forward(nearest_point.angles)[0]
    if np.linalg.norm(random_point - nearest_point.angles) > stepsize:
        steer_point = util.getSteerPoint(random_point, nearest_point.angles, stepsize)
    else:
        steer_point = np.array(random_point)
    steer_point_wk = fk.forward(steer_point)[0]
    return steer_point, steer_point_wk, nearest_point, nearest_point_wk


def choose_and_rewire(root, goal, steer_point, steer_point_wk, nearest_point, nearest_point_wk,
                      obstacles, neighbour_radius, stepsize, nodes, nodes_end):
    fk = CalculateFK()
    goal_wk = fk.forward(goal.angles)[0]
    if not util.isCollided(steer_point_wk, nearest_point_wk, obstacles):
        # neighbours = root.getNeighbours(steer_point, neighbour_radius)
        # goal_neighbours = goal.getNeighbours(steer_point, stepsize)
        neighbours = getNeighbours(steer_point, nodes, neighbour_radius)
        goal_neighbours = getNeighbours(steer_point, nodes_end, neighbour_radius)
        new_node = Node(steer_point, stepsize, parent=nearest_point, root=root)
        nodes.update({np.array2string(new_node.angles, precision=5): new_node})

        # Choose parent
        cost = new_node.cost
        for neighbour in neighbours:
            neighbour_wk = fk.forward(neighbour.angles)[0]
            if not util.isCollided(steer_point_wk, neighbour_wk, obstacles):
                new_cost = neighbour.cost + np.linalg.norm(neighbour.angles - new_node.angles)
                if new_cost < cost:
                    new_node.updateParent(neighbour, new_cost)
        new_node.parent.addChild(new_node)

        # Rewire
        for neighbour in neighbours:
            if not neighbour.equals(new_node.parent):
                neighbour_wk = fk.forward(neighbour.angles)[0]
                if util.isCollided(steer_point_wk, neighbour_wk, obstacles):
                    new_cost = new_node.cost + np.linalg.norm(neighbour.angles - new_node.angles)
                    if new_cost < neighbour.cost:
                        neighbour_parent = neighbour.parent
                        neighbour_parent.removeChild(neighbour)
                        neighbour.updateParent(new_node, new_cost)
                        new_node.addChild(neighbour)

        plt.scatter(new_node.angles[1], new_node.angles[2], s=3 ** 2, c='b')

        # Check if end reached
        if len(goal_neighbours) >= 1 and not util.isCollided(steer_point_wk, goal_wk, obstacles):
            return True, new_node, goal_neighbours[0]
    else:
        plt.scatter(steer_point[1], steer_point[2], s=3 ** 2, c='r')
    return False, None, None


def connectTrees(node, neighbour):
    start = node
    end = neighbour
    while end is not None:
        end_parent = end.parent
        end.updateParent(start, start.cost + np.linalg.norm(end.angles - start.angles))
        start.addChild(end)
        end.removeChild(start)
        start = end
        end = end_parent
