import numpy as np

class Node:
    def __init__(self, angles, dist, parent=None, root=None):
        self.parent = parent
        self.angles = angles
        self.children = {}
        if parent is not None:
            self.cost = parent.cost + dist
        else:
            self.cost = 0
        # can probably remove this
        self.root = root
        if root is not None:
            root.num_nodes += 1
        self.num_nodes = 1

    def addChild(self, child):
        self.children.update({np.array2string(child.angles, precision=5): child})

    def removeChild(self, child):
        self.children.pop(np.array2string(child.angles, precision=5), None)

    def updateParent(self, parent, new_cost):
        self.parent = parent
        self.cost = new_cost

    # Kinda scuffed, takes long ass time, might redo
    def getNearest(self, other_angles):
        minDist = np.inf
        nearest = self
        for child in self.children.values():
            candidate = child.getNearest(other_angles)
            if np.linalg.norm(candidate.angles - other_angles) < minDist:
                nearest = candidate
                minDist = np.linalg.norm(candidate.angles - other_angles)
        return nearest

    # Kinda scuffed, takes long ass time, might redo
    def getNeighbours(self, other_angles, radius):
        neighbours = []
        if np.linalg.norm(self.angles - other_angles) <= radius:
            neighbours.append(self)
        for child in self.children.values():
            child_neighbours = child.getNeighbours(other_angles, radius)
            neighbours.extend(child_neighbours)
        return neighbours

    # Kinda scuffed, takes long ass time, might redo
    def getLeaves(self):
        leaves = []
        if len(self.children) == 0:
            leaves.append(self)
        for child in self.children.values():
            leaves.extend(child.getLeaves())
        return leaves

    def getPathToRoot(self):
        path = [self.angles]
        if self.parent is not None:
            path.extend(self.parent.getPathToRoot())
        return path

    def equals(self, other):
        return np.array_equal(self.angles, other.angles)