import numpy as np
import pandas as pd
import py_rbf

data = pd.read_csv('../data/p_force_occluded.csv')

aero_nodes = np.zeros((data.shape[0], 3))
pressure_force = np.zeros((data.shape[0], 3))

aero_nodes[:, 0] = data['Points_0']
aero_nodes[:, 1] = data['Points_1']
aero_nodes[:, 2] = data['Points_2']

pressure_force[:, 0] = data['p_force_0']
pressure_force[:, 1] = data['p_force_1']
pressure_force[:, 2] = data['p_force_2']

struc_nodes = np.loadtxt('../data/beamstick.dat', skiprows=9)

n_a = aero_nodes.shape[0]
n_s = struc_nodes.shape[0]

for i in range(n_s):
    struc_nodes[i, 2] -= 1200

print(n_a, n_s)

print(np.sum(pressure_force[:, 2]))
print(np.sum(data['p_force_2']))

H_p = py_rbf.generate_transfer_matrix(aero_nodes, struc_nodes, r0=20, rbf='c2', polynomial=False)

print(H_p.shape)

F_s = py_rbf.interp_forces(pressure_force, H_p)

print(np.sum(F_s[:, 2]))

f = open('../data/loaded_beam.plt', 'w')
f.write("TITLE = \"Beamstick model\"\n")
f.write("VARIABLES = \"X\", \"Y\", \"Z\", \"p_x\", \"p_y\", \"p_z\"\n")
f.write("ZONE I={}, J=1, K=1\n".format(n_s))
for i in range(n_s):
    f.write("{} {} {} {} {} {}\n".format(struc_nodes[i][0], struc_nodes[i][1], struc_nodes[i][2], F_s[i][0], F_s[i][1], F_s[i][2]))
f.close()

f = open('../data/pressure_forces.dat', 'w')
f.write('{}\n'.format(n_s))
for i in range(n_s):
    f.write('{} {} {} \n'.format(F_s[i][0], F_s[i][1], F_s[i][2]))
f.close()

print(F_s.shape)

print('finished')

Test = 'test test'
