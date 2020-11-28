import numpy as np
import utilities as util
import matplotlib.pyplot as plt
import rrt_double
import rrt_fn
from calculateFK import CalculateFK


def rrt(map, start, goal, max_nodes, max_iter, optimize=False,
        stepsize=0.15, neighbour_radius=0.175, block_radius=25.4,
        bias_ratio=5, bias_radius=0.075):
    """
    Implement RRT algorithm in this file.
    :param map:         the map struct
    :param start:       start pose of the robot (1x6).
    :param goal:        goal pose of the robot (1x6).
    :param max_nodes
    :param max_iter
    :param optimize
    :param stepsize
    :param neighbour_radius
    :param block_radius
    :param bias_ratio
    :param bias_radius
    :return:            returns an mx6 matrix, where each row consists of the configuration of the Lynx at a point on
                        the path. The first row is start and the last row is goal. If no path is found, PATH is a 0x6
                        matrix..
    """
    fk = CalculateFK()
    obstacles = util.inflateBlocks(map.obstacles, block_radius)
    start_wk = fk.forward(start)[0]
    goal_wk = fk.forward(goal)[0]
    if util.checkStartGoal(start_wk, goal_wk, obstacles):
        print('Start or Goal in collision please redefine')
        return np.ndarray((0, 6)), 0
    nodes = {}
    path_found, root, end, num_iter, num_nodes = \
        rrt_double.rrt_double(obstacles, start, goal, max_nodes,
                              max_iter, stepsize, neighbour_radius, nodes)

    if path_found:
        old_cost = end.cost
        beacons = np.zeros([1, 6])
        iter_without_change = 0
        for i in range(num_iter, max_iter):
            if (i - num_iter) % bias_ratio == 1 and optimize:
                steer_point, steer_point_wk, nearest_point, nearest_point_wk = \
                    get_biased_sample_set(goal, root, stepsize, beacons, bias_radius)
            else:
                steer_point, steer_point_wk, nearest_point, nearest_point_wk = \
                    rrt_double.get_sample_set(goal, root, stepsize, nodes)

            if num_nodes < max_nodes:
                rrt_double.choose_and_rewire(root, end, steer_point, steer_point_wk, nearest_point,
                                             nearest_point_wk, obstacles, neighbour_radius, stepsize,
                                             nodes, {})
                num_nodes += 1
            else:
                rrt_fn.choose_and_rewire_fn(root, end, steer_point, steer_point_wk, nearest_point,
                                            nearest_point_wk, obstacles, neighbour_radius, stepsize,
                                            nodes, {})
            if optimize:
                new_cost, new_beacons = optimizePath(end, fk.forward(end.angles)[0],
                                                     end.parent, obstacles, stepsize, output=True)
                new_beacons.append(end.angles)
                if new_cost < old_cost:
                    beacons = new_beacons
                    old_cost = new_cost
                    iter_without_change = 0
                else:
                    iter_without_change += 1
            elif end.cost >= old_cost:
                iter_without_change += 1
            else:
                iter_without_change = 0

            if iter_without_change >= 100:
                break
            print("Iter: %d" % num_iter)
            num_iter += 1

        plt.scatter(root.angles[1], root.angles[2], c='y')
        plt.scatter(end.angles[1], end.angles[2], c='g')
        path = end.getPathToRoot()
        path.reverse()
        path = np.array(path)
        # path = util.populateWaypoints(path, stepsize)
        return path, end.cost
    else:
        print('Could not find path within %d iterations!' % num_iter)
        return np.ndarray((0, 6)), 0


def get_biased_sample_set(goal, root, stepsize, beacons, radius):
    fk = CalculateFK()
    upper_lim = [1.4, 1.4, 1.7, 1.7, 1.5]
    lower_lim = [-1.4, -1.2, -1.8, -1.9, -2.0]
    random_center_idx = np.random.randint(len(beacons))
    center = beacons[random_center_idx]
    random_point = util.sampleRandomRadius(lower_lim, upper_lim, goal[4], goal[5], center, radius)
    nearest_point = root.getNearest(random_point)
    nearest_point_wk = fk.forward(nearest_point.angles)[0]
    if np.linalg.norm(random_point - nearest_point.angles) > stepsize:
        steer_point = util.getSteerPoint(random_point, nearest_point.angles, stepsize)
    else:
        steer_point = np.array(random_point)
    steer_point_wk = fk.forward(steer_point)[0]
    return steer_point, steer_point_wk, nearest_point, nearest_point_wk


def optimizePath(current, current_wk, next, obstacles, stepsize, output=False):
    fk = CalculateFK()
    if next.parent is not None:
        if not util.checkVisibility(current.angles, next.parent.angles, obstacles, stepsize):
            if output:
                current.parent.removeChild(current)
                next.addChild(current)
            new_cost, new_beacons = optimizePath(next, fk.forward(next.angles)[0],
                                                 next.parent, obstacles, stepsize, output)
            cost = new_cost + np.linalg.norm(current.angles - next.angles)
            if output:
                current.updateParent(next, cost)
            beacons = [next.angles]
            beacons.extend(new_beacons)
        else:
            new_cost, new_beacons = optimizePath(current, current_wk, next.parent, obstacles, stepsize, output)
            cost = new_cost
            beacons = new_beacons
    else:
        beacons = [next.angles]
        cost = np.linalg.norm(current.angles - next.angles)
        if output:
            current.parent.removeChild(current)
            next.addChild(current)
            current.updateParent(next, cost)
    return cost, beacons
