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


class aerofoil_points():
    def __init__(self, fname):
        self.name = os.path.basename(fname)

        # read points
        self.points = np.loadtxt(fname, skiprows=3)


def plot_aerofoil(foil):
    plt.plot(foil.points[:,0], foil.points[:,1], 'b.')
    plt.axis('equal')


def plot_deformed(points):
    plt.plot(points[:,0], points[:,1], 'r.')
    plt.axis('equal')


def deform_aerofoil(foil, U, mode):
    d_points = np.zeros_like(foil.points)
    d_points[:, 0] = foil.points[:, 0]
    d_points[:, 1] = foil.points[:, 1] + 1 * U[:, mode]
    return d_points


def plot_10_modes(foil, U, n=10):
    fig = plt.figure(figsize=(15,20), dpi=100, facecolor='w', edgecolor='k')
    for i in range(n):
        ax = fig.add_subplot(int(np.ceil(n/2)), 2, i + 1)
        d_points = deform_aerofoil(foil, U, i)
        ax.plot(d_points[:, 0], d_points[:, 1])
        ax.plot(foil.points[:, 0], foil.points[:, 1])
        ax.axis('equal')

        ax.set_title('Mode {}'.format(i))


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