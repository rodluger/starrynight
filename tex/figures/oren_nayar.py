import matplotlib.pyplot as plt
import numpy as np


def R(axis=[0, 1, 0], theta=0):
    axis = np.array(axis) / np.sqrt(np.sum(np.array(axis) ** 2))
    cost = np.cos(theta)
    sint = np.sin(theta)
    return np.reshape(
        [
            cost + axis[0] * axis[0] * (1 - cost),
            axis[0] * axis[1] * (1 - cost) - axis[2] * sint,
            axis[0] * axis[2] * (1 - cost) + axis[1] * sint,
            axis[1] * axis[0] * (1 - cost) + axis[2] * sint,
            cost + axis[1] * axis[1] * (1 - cost),
            axis[1] * axis[2] * (1 - cost) - axis[0] * sint,
            axis[2] * axis[0] * (1 - cost) - axis[1] * sint,
            axis[2] * axis[1] * (1 - cost) + axis[0] * sint,
            cost + axis[2] * axis[2] * (1 - cost),
        ],
        [3, 3],
    )


res = 100
theta = np.pi / 3
b = -0.5
sig = 1.0

ct = np.cos(theta)
st = np.sin(theta)
bc = np.sqrt(1 - b ** 2)
sig2 = sig ** 2

# Grid up the surface
grid = np.linspace(-1, 1, res)
x, y = np.meshgrid(grid, grid)
x = x.reshape(-1)
y = y.reshape(-1)
z = np.sqrt(1 - x ** 2 - y ** 2)
npts = len(z)

# Lambertian intensity
ci = -bc * st * x + bc * ct * y - b * z
I_lamb = np.maximum(0, ci)

# Oren-Nayar intensity
f1 = -b / z - ci
f2 = -b / ci - z
A = 1 - 0.5 * sig2 / (sig2 + 0.33)
B = 0.45 * sig2 / (sig2 + 0.09)
f = np.maximum(0, np.minimum(f1, f2))
I_on94 = I_lamb * (A + B * f)

# Oren-Nayar intensity (exact)
zs = -b
if theta == 0:
    xs = 0
    ys = bc
else:
    xs = -bc * np.sin(theta)
    ys = bc * np.cos(theta)

# The three vectors in the observer (sky) frame
s0 = np.tile([xs, ys, zs], npts).reshape(-1, 3)  # source vector
v0 = np.tile([0, 0, 1.0], npts).reshape(-1, 3)  # observer vector
n0 = np.concatenate(
    (x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), axis=-1,
)  # normal vector

# The three vectors in the surface normal frame
# and their corresponding angles
s = np.zeros_like(s0)
v = np.zeros_like(v0)
n = np.zeros_like(n0)
theta_r = np.zeros(npts)
theta_i = np.zeros(npts)
del_phi = np.zeros(npts)
for j in range(npts):

    # Off-disk?
    if np.isnan(n0[j, 2]):
        continue

    # The surface normal
    n[j] = np.array([0, 0, 1.0])

    # Find the rotation matrix that transforms to
    # the normal frame
    axis = np.cross(n0[j], n[j])
    costhet = np.dot(n0[j], n[j])
    thet = np.arccos(costhet)
    Rn = R(axis, -thet)

    # Check that the sign is right
    Rnp = R(axis, thet)
    if Rnp.dot(n0[j])[2] > Rn.dot(n0[j])[2]:
        Rn = Rnp

    # Rotate s and v
    s[j] = Rn.dot(s0[j])
    v[j] = Rn.dot(v0[j])

    # Compute the angles
    theta_r[j] = np.arccos(np.dot(v[j], n[j]))
    theta_i[j] = np.arccos(np.dot(s[j], n[j]))
    phi_r = np.arctan2(v[j, 1], v[j, 0])
    phi_i = np.arctan2(s[j, 1], s[j, 0])
    del_phi[j] = phi_i - phi_r

# Now compute the intensity
alpha = np.maximum(theta_r, theta_i)
beta = np.minimum(theta_r, theta_i)
f = np.maximum(0, np.cos(del_phi)) * np.sin(alpha) * np.tan(beta)
I_on94_exact = I_lamb * (A + B * f)

# Plot
fig, ax = plt.subplots(1, 4)
ax[0].imshow(
    I_on94_exact.reshape(res, res),
    origin="lower",
    extent=(-1, 1, -1, 1),
    cmap="plasma",
    vmin=0,
    vmax=1.5,
)
ax[1].imshow(
    I_on94.reshape(res, res),
    origin="lower",
    extent=(-1, 1, -1, 1),
    cmap="plasma",
    vmin=0,
    vmax=1.5,
)
diff = I_on94_exact - I_on94
mx = max(np.nanmax(diff), -np.nanmin(diff))
ax[2].imshow(
    diff.reshape(res, res),
    origin="lower",
    extent=(-1, 1, -1, 1),
    vmin=-mx,
    vmax=mx,
    cmap="RdBu",
)

ax[3].plot(I_on94_exact.reshape(res, res)[:, 50])
ax[3].plot(I_on94.reshape(res, res)[:, 50])

for axis in ax.flatten():
    axis.axis("off")
plt.show()
