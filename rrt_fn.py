import numpy as np
import utilities as util
from utilities import getNeighbours
from calculateFK import CalculateFK
from node import Node
import matplotlib.pyplot as plt


def choose_and_rewire_fn(root, goal, steer_point, steer_point_wk, nearest_point, nearest_point_wk,
                         obstacles, neighbour_radius, stepsize, nodes, nodes_end):
    fk = CalculateFK()
    goal_wk = fk.forward(goal.angles)[0]
    if not util.isCollided(steer_point_wk, nearest_point_wk, obstacles):
        # neighbours = root.getNeighbours(steer_point, neighbour_radius)
        # goal_neighbours = goal.getNeighbours(steer_point, stepsize)
        neighbours = getNeighbours(steer_point, nodes, neighbour_radius)
        goal_neighbours = getNeighbours(steer_point, nodes_end, neighbour_radius)
        new_node = Node(steer_point, stepsize, parent=nearest_point)

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
        node_deleted = False
        old_parents = {}
        for neighbour in neighbours:
            if not neighbour.equals(new_node.parent):
                neighbour_wk = fk.forward(neighbour.angles)[0]
                if util.isCollided(steer_point_wk, neighbour_wk, obstacles):
                    new_cost = new_node.cost + np.linalg.norm(neighbour.angles - new_node.angles)
                    if new_cost < neighbour.cost:
                        neighbour_parent = neighbour.parent
                        old_parents.update({np.array2string(neighbour.angles, precision=5):
                                            neighbour_parent})
                        if not node_deleted and len(neighbour_parent.children) == 1:
                            neighbour_grandparent = neighbour_parent.parent
                            neighbour_grandparent.removeChild(neighbour_parent)
                            nodes.pop(np.array2string(neighbour_parent.angles, precision=5), None)
                            node_deleted = True
                        neighbour_parent.removeChild(neighbour)
                        neighbour.updateParent(new_node, new_cost)
                        new_node.addChild(neighbour)

        # Global Removal
        if not node_deleted:
            leaves = root.getLeaves()
            if len(leaves) >= 1:
                rand_leaf_idx = np.random.randint(0, len(leaves))
                leaf = leaves[rand_leaf_idx]
                leaf_parent = leaf.parent
                leaf_parent.removeChild(leaf)
                nodes.pop(np.array2string(leaf.angles, precision=5), None)
                node_deleted = True

        # If no leaves removed prune new node
        if not node_deleted:
            for child in new_node.children.values():
                child.updateParent(old_parents.get(np.array2string(child.angles, precision=5)))
                old_parents.get(np.array2string(child.angles, precision=5)).addChild(child)
            new_node_parent = new_node.parent
            new_node_parent.removeChild(new_node)
        else:
            nodes.update({np.array2string(new_node.angles, precision=5): new_node})

        if node_deleted:
            plt.scatter(new_node.angles[1], new_node.angles[2], s=3 ** 2, c='b')

        # Check if end reached
        if len(goal_neighbours) >= 1 and not util.isCollided(steer_point_wk, goal_wk, obstacles):
            return True, new_node, goal_neighbours[0]
    else:
        plt.scatter(steer_point[1], steer_point[2], s=3 ** 2, c='r')
    return False, None, None
