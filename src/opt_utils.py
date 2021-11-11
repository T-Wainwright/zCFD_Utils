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
import h5py


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


def generate_modes(data_dir, save_loc):
    # perform SVD decomposition of aerofoil modes to generate modal data and save for offline use

    # load aerodynamic mode data
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

    print('Performing SVD')

    U, S, VH = np.linalg.svd(dz, full_matrices=False)

    print('Saving data')

    # Save SVD modal information for later
    f = h5py.File(save_loc, "w")
    f.create_dataset('U', data=U)
    f.create_dataset('S', data=S)
    f.create_dataset('VH', data=VH)
    f.close()


def load_modes(mode_path):
    # Load SVD modes- saves having to rerun SVD each iteration
    f = h5py.File(mode_path, 'r')
    U = np.array(f['U'])
    S = np.array(f['S'])
    VH = np.array(f['VH'])
    return U, S, VH

def load_cage_data(cage_path):
    data = np.loadtxt(cage_path, skiprows=1)
    cage_slice = np.zeros_like(data)
    cage_slice[:, 0] = data[:, 0]
    cage_slice[:, 1] = data[:, 2]
    cage_slice[:, 1] = cage_slice[:, 1]

    return cage_slice


def plot_aerofoil(foil):
    plt.plot(foil.points[:, 0], foil.points[:, 1], 'b.')
    plt.axis('equal')


def plot_overlay(foil, cage_slice):
    plt.plot(foil.points[:, 0], foil.points[:, 1], 'b.')
    plt.plot(cage_slice[:, 0], cage_slice[:, 1])
    # plt.axis('equal')


def plot_deformed(points):
    plt.plot(points[:, 0], points[:, 1], 'r.')
    plt.axis('equal')


def deform_aerofoil(foil, U, mode, weight=1):
    d_points = np.zeros_like(foil)
    d_points[:, 0] = foil[:, 0]
    d_points[:, 1] = foil[:, 1] + weight * U[:, mode]
    d_points[:, 2] = foil[:, 2]
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
        d_points = np.zeros_like(cage_slice)
        ax = fig.add_subplot(int(np.ceil(n / 2)), 2, i + 1)
        d_points = deform_aerofoil(foil, U, i)
        deformations = d_points - foil
        cage_deformations = rbf.H @ deformations
        ax.plot(d_points[:, 0], d_points[:, 1])
        ax.plot(foil[:, 0], foil[:, 1])
        ax.plot(cage_slice[:, 0], cage_slice[:, 1], 'ro-')
        ax.plot(cage_slice[:, 0] + cage_deformations[:, 0], cage_slice[:, 1] + cage_deformations[:, 1], 'bo-')
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


data_dir = '/home/tom/Documents/University/Coding/zCFD_Utils/data/UIUC Aerofoil Library/Smoothed/'
save_loc = '/home/tom/Documents/University/Coding/zCFD_Utils/data/UIUC Aerofoil Library/Modal_data.h5'

# generate_modes(data_dir=data_dir, save_loc=save_loc)
U, S, VH = load_modes('/home/tom/Documents/University/Coding/zCFD_Utils/data/UIUC Aerofoil Library/Modal_data.h5')

cage_slice = load_cage_data('/home/tom/Documents/University/Coding/zCFD_Utils/data/domain_ordered.ctr.asa.16')
cage_slice[:, 2] = cage_slice[:, 2] * 3

naca0012 = add_z(np.loadtxt('/home/tom/Documents/University/Coding/zCFD_Utils/data/UIUC Aerofoil Library/Smoothed/NACA 0012-64.dat', skiprows=3))

# Couple control cage with naca0012 profile

rbf = py_rbf.UoB_coupling(cage_slice, naca0012)
rbf.generate_transfer_matrix(0.5, 'c2', False)

# Plot default aerofoil
plt.plot(naca0012[:, 0], naca0012[:, 1])

# perturb naca0012 by first mode, and plot result

# deformed_aerofoil = deform_aerofoil(naca0012, U, 0, weight=0.5)
deformed_aerofoil = naca0012.copy()
deformations = add_z(deformed_aerofoil - naca0012)
deformations[235:255, 1] = 0.05
deformations[45:65, 1] = 0.05
# deformations = np.zeros_like(deformations)
# deformations[:, 1] = 1

plt.plot(naca0012[:, 0] + deformations[:, 0], naca0012[:, 1] + deformations[:, 1])

# Interpolate displacements

a = rbf.H @ deformations.copy() # issue line- check out interp in py_rbf

plt.plot(cage_slice[:, 0], cage_slice[:, 1], 'ro-')
plt.plot(cage_slice[:, 0] + a[:, 0], cage_slice[:, 1] + a[:, 1], 'bo-')

plot_10_modes_cage(naca0012, U, rbf, cage_slice, n=20)

# Cast control cage along blade geometry