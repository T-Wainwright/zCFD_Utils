"""
General utilities for solver agnostic aerodynamic shape optimisation

Tom Wainwright 2021

University of Bristol

tom.wainwright@bristol.ac.uk
"""

import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import py_rbf


class aerofoil_points():
    def __init__(self, fname):
        self.name = os.path.basename(fname)

        # read points
        self.points = np.loadtxt(fname, skiprows=3)

    def get_3D_slice(self):
        return add_z(self.points)


class control_cage():
    def __init__(self, fname):
        # load from cba format
        data = np.loadtxt(fname)
        self.n_xy = int(data[0, 0])
        self.n_z = int(data[0, 1])

        self.points = data[1:, :]

    def get_slice0(self):
        return add_z(self.points[0: self.n_xy, :])


def plot_aerofoil(foil):
    plt.plot(foil.points[:, 0], foil.points[:, 1], 'b.')
    plt.axis('equal')


def plot_deformed(points):
    plt.plot(points[:, 0], points[:, 1], 'r.')
    plt.axis('equal')


def deform_aerofoil(foil, U, mode):
    d_points = np.zeros_like(foil.points)
    d_points[:, 0] = foil.points[:, 0]
    d_points[:, 1] = foil.points[:, 1] + 1 * U[:, mode]
    return d_points


def plot_10_modes(foil, U, n=10):
    fig = plt.figure(figsize=(15, 20), dpi=300, facecolor='w', edgecolor='k')
    for i in range(n):
        ax = fig.add_subplot(int(np.ceil(n / 2)), 2, i + 1)
        d_points = deform_aerofoil(foil, U, i)
        ax.plot(d_points[:, 0], d_points[:, 1])
        ax.plot(foil.points[:, 0], foil.points[:, 1])
        ax.axis('equal')
        ax.set_xlabel('x/c', fontsize=12)
        ax.set_ylabel('z', fontsize=12)

        ax.set_title('Mode {}'.format(i), fontsize=18)

    fig.tight_layout()


def plot_10_modes_cage(foil, U, rbf, cage_slice, n=10):
    fig = plt.figure(figsize=(15, 20), dpi=300, facecolor='w', edgecolor='k')
    for i in range(n):
        ax = fig.add_subplot(int(np.ceil(n / 2)), 2, i + 1)
        d_points = deform_aerofoil(foil, U, i)
        deformations = add_z(d_points - foil.points)
        cage_deformations = np.matmul(rbf.H, deformations)
        ax.plot(d_points[:, 0], d_points[:, 1])
        ax.plot(foil.points[:, 0], foil.points[:, 1])
        ax.plot(cage_slice[:, 0], cage_slice[:,1])
        ax.plot(cage_slice[:,0] + cage_deformations[:,0], cage_slice[:,0] + cage_deformations[:,1])
        ax.axis('equal')
        ax.set_xlabel('x/c', fontsize=12)
        ax.set_ylabel('z', fontsize=12)

        ax.set_title('Mode {}'.format(i), fontsize=18)

    fig.tight_layout()





def load_control_cage(fname):
    cage_nodes = np.loadtxt(fname, skiprows=3)
    return cage_nodes


def add_z(points):  
    dim1 = len(points[:, 0])
    points_3D = np.zeros([dim1, 3])
    for i in range(dim1):
        points_3D[i, :] = [points[i, 0], points[i, 1], 0]

    return points_3D


# load aerodynamic mode data
data_dir = '/home/tom/Documents/University/Coding/zCFD_Utils/data/UIUC Aerofoil Library/Smoothed/'
aerofoil_names = glob.glob(data_dir + '*')

aerofoils = {}

for n in aerofoil_names:
    name = os.path.basename(n)
    aerofoils[name] = aerofoil_points(n)

foil_keys = list(aerofoils.keys())

n_foils = len(aerofoils)
n_points = len(aerofoils[next(iter(aerofoils))].points[:, 0])
m_def = int((n_foils * (n_foils - 1)) / 2)
# create dZ matrix:
dz = np.zeros([n_points, m_def])
n = 0

for i in range(n_foils):
    for j in range(i + 1, n_foils):
        dz[:, n] = aerofoils[foil_keys[i]].points[:, 1] - aerofoils[foil_keys[j]].points[:, 1]
        n += 1

# Perform SVD to get mode shapes and energies

U, S, VH = np.linalg.svd(dz, full_matrices=False)

# Load control cage

# cage_nodes = control_cage('/home/tom/Documents/University/Coding/zCFD_Utils/data/control_cage.cba')
# cage_slice = cage_nodes.get_slice0()
x = np.array([0.8, 0.5, 0.3, 0.1, 0, -0.1, 0, 0.1, 0.3, 0.5, 0.8])
y = np.array([1, 1, 1, 1, 0.5, 0, -0.5, -1, -1, -1, -1])
cage_slice = np.transpose(np.vstack([x,y]))
cage_slice = add_z(cage_slice)
# cage_slice = cage_slice / 4
cage_slice[:, 1] = cage_slice[:, 1] * 0.12

# load naca0012 nodes
naca0012_nodes = aerofoils['NACA 0012-64.dat'].points

# Couple control cage 0th nodes with naca0012 profile:

rbf = py_rbf.UoB_coupling(cage_slice, aerofoils['NACA 0012-64.dat'].get_3D_slice())
rbf.generate_transfer_matrix(20, 'c2', False)

# perturb naca0012 by first mode, and print result

deformed_aerofoil = deform_aerofoil(aerofoils['NACA 0012-64.dat'], U, 0)
deformations = add_z(deformed_aerofoil - aerofoils['NACA 0012-64.dat'].points)

a = np.matmul(rbf.H, deformations) # issue line- check out interp in py_rbf

# plot cage deformations
plot_10_modes_cage(aerofoils['NACA 0012-64.dat'], U, rbf, cage_slice)
