import numpy as np
import matplotlib.pyplot as plt
import multiscale

X = np.loadtxt(
    '/home/tom/Documents/University/Coding/meshdef/examples/2_xyz/surface.xyz', skiprows=1)
V = np.loadtxt(
    '/home/tom/Documents/University/Coding/meshdef/examples/2_xyz/volume.xyz', skiprows=1)
dX = np.loadtxt(
    '/home/tom/Documents/University/Coding/meshdef/examples/2_xyz/displacements.xyz', skiprows=1)


t = np.deg2rad(10)
R = np.array([[np.cos(t), -np.sin(t), 0],
             [np.sin(t), np.cos(t), 0], [0, 0, 1]])

dX = np.zeros((X.shape[0], 6))

# dX = X@R - X

M = multiscale.multiscale(X, 13, 4)
M.sample_control_points()

active_list = M.get_active_list()
radii = M.get_radii()

# f = open('/home/tom/Documents/University/Coding/cases/IEA_15MW/active_list.dat', "w")
# for p in active_list:
#     f.write('{}\n'.format(p))
# f.close()

# f = open('/home/tom/Documents/University/Coding/cases/IEA_15MW/radii.dat', "w")
# for p in active_list:
#     f.write('{}\n'.format(p))
f.close()
M.multiscale_solve(dX)
M.preprocess_V(V)
M.multiscale_transfer()

dV = M.get_dV()

# plt.plot(X[:, 0], X[:, 1])
# plt.plot(X[:, 0] + dX[:, 0], X[:, 1] + dX[:, 1])

# plt.plot(V[:, 0], V[:, 1])
# plt.plot(V[:, 0] + dV[:, 0], V[:, 1] + dV[:, 1])

print(M.get_radii())
print(M.get_active_list())
print(M.get_a())
print(dV)
# print(dv)
plt.show()
