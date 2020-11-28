import numpy as np
from detectCollision import detectCollision
from calculateFK import calculateFK
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sampleRandom(lower_lim, upper_lim, wrist_rotation, end_effector_gap):
    point = [np.random.uniform(lower_lim[i], upper_lim[i]) for i in range(4)]
    point.append(wrist_rotation)
    point.append(end_effector_gap)
    return point

def sampleRandomRadius(lower_lim, upper_lim, wrist_rotation, end_effector_gap, center, radius):
    rng = np.random.default_rng()
    X = rng.normal(size=(1, 4))
    U = rng.random((1, 1))
    point = (radius * U ** (1 / 4) / np.sqrt(np.sum(X ** 2, 1, keepdims=True)) * X) + center[:4]
    point = np.clip(point, lower_lim[:4], upper_lim[:4]).squeeze().tolist()
    point.append(wrist_rotation)
    point.append(end_effector_gap)
    return point

def getSteerPoint(randNode, nearestNode, stepSize):
    delta = randNode - nearestNode
    steer = delta / np.linalg.norm(delta)
    return nearestNode + steer * stepSize

def checkStartGoal(start, goal, boxes):
    isCollided = False
    for box in boxes:
        isCollided = isCollided or np.any(detectCollision(start, start, box))
        isCollided = isCollided or np.any(detectCollision(goal, goal, box))
        if isCollided:
            return isCollided
    return isCollided

def isCollided(pts1, pts2, boxes):
    isCollided = False
    next_joint_idx = [1, 2, 3, 4, 5]
    link_pts_1 = pts1[next_joint_idx, :]
    link_pts_2 = pts2[next_joint_idx, :]
    # Check self collision
    rs = link_pts_2 - pts2[:5, :]
    for i in range(len(rs) - 1):
        p = pts2[i]
        r = rs[i]
        for j in range(i + 1, len(rs)):
            q = pts2[j]
            s = rs[j]
            rxs = np.cross(r, s).sum()
            qpxr = np.cross(q-p, r).sum()
            if rxs == 0 and qpxr == 0 and np.dot(s, r) < 0:
                t0 = np.dot(q-p, r) / np.dot(r, r)
                t1 = np.dot(q+s-p, r) / np.dot(r, r)
                if not (t1 > 1 or t0 < 0):
                    return True
            elif rxs != 0 and i < 2:
                t = np.cross(q - p, s).sum() / np.cross(r, s).sum()
                u = np.cross(q - p, r).sum() / np.cross(r, s).sum()
                if 0 < t < 1 and 0 < u < 1:
                    return True
    # Check external collision
    pts1 = np.vstack((pts1, pts1[:5, :], link_pts_2))
    pts2 = np.vstack((pts2, link_pts_1, pts2[:5, :]))
    for box in boxes:
        isCollided = np.any(detectCollision(pts1, pts2, box))
        if isCollided:
            return isCollided
    return isCollided

def checkVisibility(q1, q2, boxes, stepsize):
    # Discrete line between q1 and q2 in config space
    # check consecutive point on that line for collision
    fk = calculateFK()
    num_samples = int(np.ceil(np.linalg.norm(q2 - q1) / (stepsize / 2)))
    discrete_pts = [np.linspace(q1[i], q2[i], num_samples).tolist()
                    for i in range(len(q1))]
    discrete_pts = np.array(discrete_pts)
    discrete_pts = discrete_pts.T
    discrete_fk = fk.forward(discrete_pts[0])[0]
    collided = False
    for i in range(1, len(discrete_pts)):
        next_fk = fk.forward(discrete_pts[i])[0]
        collided = collided or isCollided(discrete_fk, next_fk, boxes)
        discrete_fk = next_fk
    return not collided

def populateWaypoints(path, stepsize):
    if len(path) > 0:
        waypoints = path[0]
        q1 = path[0]
        for i in range(1, len(path)):
            q2 = path[i]
            num_samples = int(np.floor(np.linalg.norm(q2 - q1) / stepsize))
            discrete_pts = [np.linspace(q1[i], q2[i], num_samples).tolist()
                            for i in range(len(q1))]
            discrete_pts = np.array(discrete_pts)
            discrete_pts = discrete_pts.T
            waypoints = np.vstack((waypoints, discrete_pts))
            q1 = q2
        print(waypoints.shape)
        return waypoints
    return []

def inflateBlocks(blocks, radius):
    modifier = np.array([-radius, -radius, -radius, radius, radius, radius])
    return np.array([modifier + block for block in blocks])

def getNeighbours(other_angles, nodes, radius):
    neighbours = []
    for node in nodes.values():
        if np.linalg.norm(node.angles - other_angles) <= radius:
            neighbours.append(node)
    return neighbours

def getNearest(other_angles, nodes):
    minDist = np.inf
    nearest = None
    for node in nodes.values():
        if np.linalg.norm(node.angles - other_angles) < minDist:
            nearest = node
            minDist = np.linalg.norm(node.angles - other_angles)
    return nearest

def reachedTarget(pos, target, radius):
    if np.linalg.norm(pos[:5] - target[:5]) < radius:
        return True
    else:
        return False
