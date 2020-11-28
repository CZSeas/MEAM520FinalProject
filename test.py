import visualize
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from calculateFK import calculateFK
from loadmap import loadmap
from copy import deepcopy
import numpy as np
from rrt import rrt
from astar import Astar

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d([-300.0, 300.0])
ax.set_ylim3d([-300.0, 300.0])
ax.set_zlim3d([-100.0, 300.0])

map_struct = loadmap("maps/final.txt")
obstacles = map_struct.obstacles
start = np.array([0, 0, 0, 0, 0, 0])
goal = np.array([-1, 0, 1, 0, 0, 0])

max_nodes = 5000
max_iter = 3000

# path, cost = rrt(deepcopy(map_struct), deepcopy(start), deepcopy(goal), max_nodes, max_iter,
#                  stepsize=0.1, neighbour_radius=0.15, bias_ratio=5, bias_radius=0.075,
#                  optimize=True)

path = Astar(deepcopy(map_struct), deepcopy(start), deepcopy(goal))

fk = calculateFK()
start_pos = fk.forward(path[0])[0]
line = ax.plot(start_pos[:, 0], start_pos[:, 1], start_pos[:, 2])[0]

visualize.plot_obstacles(ax, obstacles)

arm_ani = animation.FuncAnimation(fig, visualize.animate, fargs=(path, line), interval=20, blit=False)


plt.show()
