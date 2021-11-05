"""
ATOM Wrapper to combine ATOM functionality with generic RBF coupling 

ATOM (Aeroelastic Turbine Optimisation Method) by Terence Macquart and Sam Scott

Tom Wainwright 2021

University of Bristol

tom.wainwright@bristol.ac.uk
"""

import numpy as np
import scipy.io
import h5py
import matplotlib.pyplot as plt


class atom_struct():
    # Wrapper to process ATOM structural data into a useful format for RBF work
    def __init__(self, bladeFE):
        print('Loading FE structural data from {}'.format(fname))
        self.BladeFE = scipy.io.loadmat(fname)['Blade_FE']

        temp = self.BladeFE['BeamAxis_123'][0][0]

        self.struct_nodes = np.zeros_like(temp)

        # Convert ATOM coordinate system into CFD frame

        self.struct_nodes[:, 0] = temp[:, 1]
        self.struct_nodes[:, 1] = temp[:, 2]
        self.struct_nodes[:, 2] = temp[:, 0]

        self.n_s = self.struct_nodes.shape[0]

        print('Adding FSI ribs')
        self.add_struct_ribs(1)

    def add_struct_ribs(self, rib_length):
        self.rib_nodes = np.zeros((self.n_s * 4, 3))
        N_offset = [1 * rib_length, 0, 0]
        S_offset = [-1 * rib_length, 0, 0]
        E_offset = [0, 1 * rib_length, 0]
        W_offset = [0, -1 * rib_length, 0]

        for i in range(self.n_s):
            # print(i)
            self.rib_nodes[i * 4 + 0, :] = self.struct_nodes[i, :] + N_offset
            self.rib_nodes[i * 4 + 1, :] = self.struct_nodes[i, :] + S_offset
            self.rib_nodes[i * 4 + 2, :] = self.struct_nodes[i, :] + E_offset
            self.rib_nodes[i * 4 + 3, :] = self.struct_nodes[i, :] + W_offset

    def plot_struct(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot(self.struct_nodes[:, 0], self.struct_nodes[:, 1], self.struct_nodes[:, 2], 'b.')
        ax.plot(self.rib_nodes[:, 0], self.rib_nodes[:, 1], self.rib_nodes[:, 2], 'r.')
        set_axes_equal(ax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    def load_modes(self, fname):
        print('Loading Modal data from {}'.format(fname))
        self.ModalStruct = scipy.io.loadmat(fname)['Modal_Struct']

        self.nEigval = self.ModalStruct['nEigval'][0][0][0][0]
        self.Eigvec = self.ModalStruct['Eigvec'][0][0]
        self.KmodalInv = self.ModalStruct['KmodalInv'][0][0]

    def deform_struct(self, F_s):
        # Rearrange F_s to match ATOM coordinate system
        F_s_atom = np.zeros_like(F_s)
        F_s_atom[:, 0] = F_s[:, 1]
        F_s_atom[:, 1] = F_s[:, 2]
        F_s_atom[:, 2] = F_s[:, 0]

        # Add means for factoring in moments here

        # Reshape F_s
        F_s_atom = np.concatenate((F_s_atom, np.zeros_like(F_s_atom)), axis=1)
        F_s_atom = F_s_atom.flatten(order='F')

        # Convert Nodal forces to modal forces
        FModal = np.matmul(np.transpose(self.Eigvec[:, 0:self.nEigval]), F_s_atom[6:])

        QDisp = np.matmul(self.KmodalInv, FModal)
        Disp = np.zeros([self.Eigvec.shape[0]])

        for imode in range(self.nEigval):
            Disp = Disp + QDisp[imode] * self.Eigvec[:, imode]

        Disp = np.concatenate((np.zeros(6), Disp))
        Disp = np.reshape(Disp, (seselflf.n_s, 6))

        return(Disp[:, :3] * 1.0e-0)

    def generate_test_deformation(self):
        # Return a parabolically deformed and twisted blade without the need for force application
        Disp = np.zeros((self.n_s, 6))
        for i in range(self.n_s):
            # [x_disp, y_disp, z_disp, x_rot, y_rot, z_rot]
            # Disp[i, 0] = self.struct_nodes[i, 2] ** 2 * 1e-4 * 4
            Disp[i, 1] = 0
            Disp[i, 2] = 0
            Disp[i, 3] = 0
            Disp[i, 4] = 0
            Disp[i, 5] = 0

        return (Disp)

    def deform_ribs(self, disp):
        rib_disps = np.zeros((self.n_s * 4, 3))
        for i in range(self.n_s):
            # rib_disps[i * 4 + 0, :] = disp[i, :3] + np.matmul(np.array([1, 0, 0]), np.matmul(R_x(disp[i, 3]), np.matmul(R_y(disp[i, 4]), R_z(disp[i, 5]))))
            # rib_disps[i * 4 + 1, :] = disp[i, :3] + np.matmul(np.array([-1, 0, 0]), np.matmul(R_x(disp[i, 3]), np.matmul(R_y(disp[i, 4]), R_z(disp[i, 5]))))
            # rib_disps[i * 4 + 2, :] = disp[i, :3] + np.matmul(np.array([0, 1, 0]), np.matmul(R_x(disp[i, 3]), np.matmul(R_y(disp[i, 4]), R_z(disp[i, 5]))))
            # rib_disps[i * 4 + 3, :] = disp[i, :3] + np.matmul(np.array([0, -1, 0]), np.matmul(R_x(disp[i, 3]), np.matmul(R_y(disp[i, 4]), R_z(disp[i, 5]))))

            rib_disps[i * 4 + 0, :] = (np.matmul(np.array([1, 0, 0]), R_z(np.pi / 2)) + self.struct_nodes[i, :])
            rib_disps[i * 4 + 1, :] = (np.matmul(np.array([-1, 0, 0]), R_z(np.pi / 2)) + self.struct_nodes[i, :])
            rib_disps[i * 4 + 2, :] = (np.matmul(np.array([0, 1, 0]), R_z(np.pi / 2)) + self.struct_nodes[i, :])
            rib_disps[i * 4 + 3, :] = (np.matmul(np.array([0, -1, 0]), R_z(np.pi / 2)) + self.struct_nodes[i, :])

        return rib_disps

    def plot_deformed_struct(self, disp, rib_disp):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot(self.struct_nodes[:, 0], self.struct_nodes[:, 1], self.struct_nodes[:, 2], 'b.')
        ax.plot(self.rib_nodes[:, 0], self.rib_nodes[:, 1], self.rib_nodes[:, 2], 'b.')
        ax.plot(self.struct_nodes[:, 0] + disp[:, 0], self.struct_nodes[:, 1] + disp[:, 1], self.struct_nodes[:, 2] + disp[:, 2], 'r.')
        ax.plot(self.rib_nodes[:, 0] + rib_disp[:, 0], self.rib_nodes[:, 1] + rib_disp[:, 1], self.rib_nodes[:, 2] + rib_disp[:, 2], 'r.')
        ax.plot(disp[:, 0], disp[:, 1], disp[:, 2], 'g.')
        ax.plot(rib_disp[:, 0], rib_disp[:, 1], rib_disp[:, 2], 'g.')
        set_axes_equal(ax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(['original', 'deformed', 'deformation'])
        plt.show()

    def write_deformed_struct(self, U_s):
        # Dump out record of deformed structure
        f = open("deformed_struct.dat", 'w')
        for i in range(self.n_s):
            f.write('{} \t {} \t {}\n'.format(U_s[i, 0], U_s[i, 1], U_s[i, 2]))
        f.close()


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
    ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def R_z(a):
    R_z = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return R_z


def R_y(b):
    R_y = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    return R_y


def R_x(c):
    R_x = np.array([[1, 0, 0], [0, np.cos(c), -np.sin(c)], [0, np.sin(c), np.cos(c)]])
    return R_x
