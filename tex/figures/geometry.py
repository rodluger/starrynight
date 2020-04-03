from code import visualize
import numpy as np
import matplotlib.pyplot as plt

# DEBUG
plt.switch_backend("MacOSX")
b, theta, bo, ro = (
    0.5488316824842527,
    4.03591586925189,
    0.34988513192814663,
    0.7753986686719786,
)
fig, ax = visualize(b, theta, bo, ro)
plt.show()
quit()

b = 0.5
theta = 75 * np.pi / 180
bo = 0.5
ro = 0.7
fig, ax = visualize(b, theta, bo, ro)
for axis in ax:
    axis.set_rasterization_zorder(0)
fig.savefig("geometry.pdf", bbox_inches="tight")
