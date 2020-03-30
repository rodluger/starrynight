from code import visualize
import numpy as np
import matplotlib.pyplot as plt


b = 0.5
theta = 75 * np.pi / 180
bo = 0.5
ro = 0.7
fig, ax = visualize(b, theta, bo, ro)
fig.savefig("geometry.pdf", bbox_inches="tight")
