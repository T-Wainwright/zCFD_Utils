import numpy as np
import py_rbf

aero_nodes = np.loadtxt('../data/aero_nodes.dat', skiprows=1)
struct_nodes = np.loadtxt('../data/beamstick.dat', skiprows=1)

H = py_rbf.generate_transfer_matrix(aero_nodes, struct_nodes, r0=20, polynomial=False)

U_s = np.loadtxt('../data/deformed.dat', skiprows=1)

U_a = py_rbf.interp_displacements(U_s, H)
