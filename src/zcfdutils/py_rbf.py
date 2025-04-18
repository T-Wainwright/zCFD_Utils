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
import zcfdutils.Wrappers.ATOM_Wrapper
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KDTree


# Functions
class UoB_coupling():
    # Class to contain the meat and potatoes of rbf coupling- agnostic of any other data
    def __init__(self, mesh1, mesh2):
        # Coupling scheme will couple mesh 1 to mesh 2 in that direction- inverting an n2 x n2 matrix
        self.mesh1_nodes = mesh1
        self.mesh2_nodes = mesh2

        self.n1 = len(mesh1[:, 0])
        self.n2 = len(mesh2[:, 0])

    def generate_transfer_matrix(self, r0, rbf='c2', polynomial=True):
        # Will generate H12 matrix
        self.H = generate_transfer_matrix(
            self.mesh1_nodes, self.mesh2_nodes, r0, rbf, polynomial)

    def agglomorate(self, nclusters):
        cluster = MiniBatchKMeans(n_clusters=nclusters)
        self.agglomoration = cluster.fit(self.mesh2_nodes)

        u_labels = np.unique(self.agglomoration.labels_)
        fig = plt.figure()
        ax = fig.add_subplot()
        for i in u_labels:
            ax.scatter(self.mesh2_nodes[self.agglomoration.labels_ == i, 2],
                       self.mesh2_nodes[self.agglomoration.labels_ == i, 1])
            print(i)
        fig.savefig('test.png', format='png')

    def idw_mapping_12(self, r0):
        # maps mesh 1 to 2 using inverse distance:
        self.mapped_nodes_12 = {}
        for i in range(self.n1):
            r_t = 0
            self.mapped_nodes_12[i] = {}
            for j in range(self.n2):
                rad = norm(self.mesh1_nodes[i, :] - self.mesh2_nodes[j, :])
                if rad < r0 ** 2:
                    r = np.sqrt(rad)
                    self.mapped_nodes_12[i][j] = r
                    r_t += r

            for k in self.mapped_nodes_12[i].keys():
                self.mapped_nodes_12[i][k] = self.mapped_nodes_12[i][k] / r_t

    def idw_mapping_21(self, r0):
        # maps mesh 1 to 2 using inverse distance:
        self.mapped_nodes_21 = {}
        for j in range(self.n2):
            r_t = 0
            self.mapped_nodes_21[j] = {}
            for i in range(self.n1):
                rad = norm(self.mesh1_nodes[i, :] - self.mesh2_nodes[j, :])
                if rad < r0 ** 2:
                    r = np.sqrt(rad)
                    self.mapped_nodes_21[j][i] = r
                    r_t += r

            for k in self.mapped_nodes_21[j].keys():
                self.mapped_nodes_21[j][k] = self.mapped_nodes_21[j][k] / r_t

    def idw_interp_12(self, f):
        f_i = np.zeros((self.n2, 3))
        for i in self.mapped_nodes_12.keys():
            for j in self.mapped_nodes_12[i].keys():
                f_i[i, :] += f[j, :] * self.mapped_nodes_12[i][j]

        return f_i

    def idw_interp_21(self, f):
        f_i = np.zeros((self.n1, 3))
        for i in self.mapped_nodes_21:
            for j in self.mapped_nodes_21[i]:
                f_i[j, :] += f[i, :] * self.mapped_nodes_21[i][j]

        return f_i

    def rbf_interp_12(self, U1):
        U2 = rbf_interp(U1, self.H)
        return U2

    def rbf_interp_21(self, U2):
        U1 = rbf_interp(U2, self.H.T)
        return U1


def generate_transfer_matrix(mesh1, mesh2, r0, rbf='c2', polynomial=True):
    # returns- H: (n1,n2) full transfer matrix between mesh2 and mesh1 - inverting an n2 x n2 matrix
    # Reference DOI: 10.1002/nme.2219
    n_1 = len(mesh1[:, 0])
    n_2 = len(mesh2[:, 0])

    switcher = {'c0': c0, 'c2': c2, 'c4': c4, 'c6': c6}
    rbf = switcher.get(rbf)

    # preallocate matrices
    if polynomial:
        A_12 = np.zeros((n_1, n_2 + 4))
        P_2 = np.ones((4, n_2))
    else:
        A_12 = np.zeros((n_1, n_2))

    M_22 = np.zeros((n_2, n_2))

    print('generate block M')

    # Generate block matrix M (and P if polynomial) equations 11 and 12
    for i in range(n_2):
        for j in range(n_2):
            rad = norm(mesh2[i, :] - mesh2[j, :]) / (r0 ** 2)
            # rad = anorm(mesh2[i, :] - mesh2[j, :]) / r0
            if rad <= 1.0:
                M_22[i][j] = rbf(np.sqrt(rad))
        if polynomial:
            P_2[1:, i] = mesh2[i]
        if i % 1000 == 0:
            print(i)

    print('generate block A_12')

    # Generate A_12 matrix- equation 13
    for i in range(n_1):
        for j in range(n_2):
            rad = norm(mesh1[i, :] - mesh2[j, :]) / (r0 ** 2)
            # rad = anorm(mesh1[i, :] - mesh2[j, :]) / r0
            if rad <= 1.0:
                if polynomial:
                    A_12[i][j + 4] = rbf(np.sqrt(rad))
                else:
                    A_12[i][j] = rbf(np.sqrt(rad))
        if polynomial:
            A_12[i][1:4] = mesh1[i]
            A_12[i][0] = 1
        if i % 1000 == 0:
            print(i)

    M_inv = np.linalg.inv(M_22)

    if polynomial:
        # Equations 21 and 22 in Allen and Rendall
        M_p = np.linalg.pinv((P_2 @ M_inv) @ P_2.T)

        Top = (M_p @ P_2) @ M_inv

        Bottom = M_inv - (((M_inv @ P_2.T) @ M_p) @ P_2) @ M_inv

        B = np.concatenate((Top, Bottom))

        H = A_12 @ B
    else:
        H = A_12 @ M_inv
        print(A_12.shape)
        print(M_inv.shape)
    return H


def interp_displacements(U_s, H):
    # Interpolate structural displacements U_s to aerodynamic surface nodes
    U_a = np.zeros((H.shape[0], U_s.shape[1]))

    for i in range(U_s.shape[1]):
        U_a[:, i] = H @ U_s[:, i]
    return U_a


def interp_forces(F_a, H):
    # Interpolate aerodynamic forces F_a to structural nodes
    F_s = np.zeros((H.shape[1], F_a.shape[1]))

    for i in range(F_a.shape[1]):
        F_s[:, i] = H.T @ F_a[:, i]
    return F_s


def rbf_interp(U, H):
    return H @ U


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


def anorm(vec):
    bias = [1, 1, 0.1]
    r = np.sqrt((vec[0] * bias[0])**2 + (vec[1] * bias[1])
                ** 2 + (vec[2] * bias[2])**2)
    return r


def norm(vec):
    r = vec[0]**2 + vec[1]**2 + vec[2] ** 2
    return r

class IDWMapper():
    def __init__(self, source, target) -> None:
        self.source = source
        self.target = target

        self.ns = source.shape[0]
        self.nt = target.shape[0]

    def map(self, source_data, n=4):
        target_data = np.zeros_like(self.target)
        X_tree = KDTree(self.target, leaf_size = 10, metric='euclidean')
        for i in range(self.ns):
            dist, ind = X_tree.query([self.source[i]], k=n)    
            sum_of_distances = sum(dist[0])
            for index, rad in zip(ind[0], dist[0]):
                target_data[index, :] += source_data[i, :] * rad / sum_of_distances

        return target_data