"""
Solver and Physics agnostic RBF coupling utilities

Used for interpolating data between 2 local meshes:
FSI:
-Aero forces -> structural model
-Structural displacements -> Aero surface
Optimisation:
-Modal Deformation -> control cage
-Control cage deformations -> Aero surface

Tom Wainwright 2021

University of Bristol

tom.wainwright@bristol.ac.uk
"""
import numpy as np
import scipy.io
import h5py
import matplotlib.pyplot as plt
import ATOM_Wrapper


# Functions
class UoB_coupling():
    # Class to contain the meat and potatoes of rbf coupling- agnostic of any other data
    def __init__(self, mesh1, mesh2):
        # Coupling scheme will couple mesh 1 to mesh 2 in that direction- inverting an n2 x n2 matrix
        self.mesh1_nodes = mesh1
        self.mesh2_nodes = mesh2

        self.n1 = len(mesh1[:, 0])
        self.n2 = len(mesh2[:, 2])

    def generate_transfer_matrix(self, r0, rbf='c2', polynomial=True):
        self.H = generate_transfer_matrix(self.mesh1_nodes, self.mesh2_nodes, r0, rbf, polynomial)

    def interp_12(self, U1):
        U2 = rbf_interp(U1, self.H)
        return U2

    def interp_21(self, U2):
        U1 = rbf(U2, np.transpose(self.H))
        return U1


def generate_transfer_matrix(mesh1, mesh2, r0, rbf='c2', polynomial=True):
    # returns- H: (n1,n2) full transfer matrix between mesh2 and mesh1 - inverting an n2 x n2 matrix
    # Reference DOI: 10.1002/nme.2219
    n_a = len(mesh1[:, 0])
    # n_a = mesh1[:,0].size
    n_s = len(mesh2[:, 0])

    switcher = {'c0': c0, 'c2': c2, 'c4': c4, 'c6': c6}
    rbf = switcher.get(rbf)

    # preallocate matrices    
    if polynomial:
        A_as = np.zeros((n_a, n_s + 4))
        P_s = np.ones((4, n_s))
    else:
        A_as = np.zeros((n_a, n_s))

    M_ss = np.zeros((n_s, n_s))

    print('generate block M')

    # Generate block matrix M (and P if polynomial) equations 11 and 12
    for i in range(n_s):
        for j in range(n_s):
            rad = (np.linalg.norm((mesh2[i] - mesh2[j]))) / r0
            if rad <= 1.0:
                M_ss[i][j] = rbf(rad)
        if polynomial:
            P_s[1:, i] = mesh2[i]
        if i % 1000 == 0:
            print(i)

    print('generate block A_as')

    # Generate A_as matrix- equation 13
    for i in range(n_a):
        for j in range(n_s):
            rad = np.linalg.norm((mesh1[i] - mesh2[j])) / r0
            if rad <= 1.0:
                if polynomial:
                    A_as[i][j + 4] = rbf(rad)
                else:
                    A_as[i][j] = rbf(rad)
        if polynomial:
            A_as[i][1:4] = mesh1[i]
            A_as[i][0] = 1
        if i % 1000 == 0:
            print(i)

    M_inv = np.linalg.pinv(M_ss)
    if polynomial:
        # Equations 21 and 22 in Allen and Rendall
        M_p = np.linalg.pinv(np.matmul(np.matmul(P_s, M_inv), np.transpose(P_s)))

        Top = np.matmul(np.matmul(M_p, P_s), M_inv)
        Bottom = M_inv - np.matmul(np.matmul(np.matmul(np.matmul(M_inv, np.transpose(P_s)), M_p), P_s), M_inv)

        B = np.concatenate((Top, Bottom))

        H = np.matmul(A_as, B)
    else:
        H = np.matmul(A_as, M_inv)
    return H


def interp_displacements(U_s, H):
    # Interpolate structural displacements U_s to aerodynamic surface nodes
    U_a = np.zeros((H.shape[0], U_s.shape[1]))

    for i in range(U_s.shape[1]):
        U_a[:, i] = np.matmul(H, U_s[:, i])
    return U_a


def interp_forces(F_a, H):
    # Interpolate aerodynamic forces F_a to structural nodes
    F_s = np.zeros((H.shape[1], F_a.shape[1]))

    for i in range(F_a.shape[1]):
        F_s[:, i] = np.matmul(np.transpose(H), F_a[:, i])
    return F_s


def rbf_interp(U1, H):
    # perform matrix multiplacation to interpolate U1 to U2
    U2 = np.zeros((H.shape[1], U1.shape[1]))
    for i in range(U1.shape[1]):
        U2[:, i] = np.matmul(H, U2[:, i])
    return U2


def c0(r):
    psi = (1 - r)**2
    return psi


def c2(r):
    psi = ((1 - r)**4) * (4 * r + 1)
    return psi


def c4(r):
    psi = ((1 - r)**6) * (35 * r**2 + 18 * r + 3)
    return psi


def c6(r):
    psi = ((1 - r)**8) * (32 * r**3 + 25 * r**2 + 8 * r + 1)
    return psi


