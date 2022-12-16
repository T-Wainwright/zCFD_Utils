import numpy as np
from scipy import sparse
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import time


class MultiScale():
    def __init__(self, X, nb, r) -> None:
        self.X = X
        self.np = self.X.shape[0]
        self.nb = int(np.ceil(nb * self.np))
        self.r = r

    def sequence_control_points(self, parent_weighting=True):
        # Create active lists
        active_list = []
        inactive_list = [i for i in range(self.np)]
        sep_dist = [1e10 for i in range(self.np)]
        radii = [self.r for i in range(self.np)]
        parent = [0 for i in range(self.np)]
        base_set = []
        remaining_set = []

        n_active = 0

        # Grab first control point
        active_node = inactive_list[0]
        # sep_dist[active_node] = -1e10
        active_list.append(active_node)
        inactive_list.remove(active_node)
        base_set.append(active_node)

        n_active += 1

        # build KD Tree to speed up radius searches

        X_tree = KDTree(self.X, leaf_size=10, metric='euclidean')

        # Cycle over remaining points

        while n_active < self.np:
            ind, dist = X_tree.query_radius(
                [self.X[active_node]], sep_dist[active_node], return_distance=True)
            # dist, ind = X_tree.query(
            #     [self.X[active_node]], self.np)

            sep_dist[active_node] = -1e-10

            for i, p in enumerate(ind[0]):
                if dist[0][i] < sep_dist[p]:
                    sep_dist[p] = dist[0][i]
                    parent[p] = active_node

            sep_dist_temp = sep_dist.copy()

            if parent_weighting:
                # if searching base set, find the node with the max separation from the last added node
                if n_active < self.nb:
                    children = [parent.count(i) for i in active_list]
                    iMax = np.argmax(children)
                    for i, p in enumerate(parent):
                        if p != active_list[iMax]:
                            sep_dist_temp[i] = -1

                    active_node = np.argmax(sep_dist_temp)

                # if searching remaining set, find the node with max separation from all nodes
                else:
                    active_node = np.argmax(sep_dist)
            else:
                active_node = np.argmax(sep_dist)

            # find next active node- one with furthest distance from last activated point
            active_list.append(active_node)
            inactive_list.remove(active_node)

            if n_active < self.nb:
                radii[active_node] = self.r
                base_set.append(active_node)
            else:
                radii[active_node] = sep_dist[active_node]
                remaining_set.append(active_node)

            # sep_dist[active_node] = -1e10

            n_active += 1

        self.active_list = active_list
        self.radii = radii
        self.base_set = base_set
        self.remaining_set = remaining_set

    def multiscale_solve(self, dX):
        self.dX = dX.copy()
        self.reorder()
        self.generate_b_KD()
        self.generate_r_KD()
        self.generate_LCRS_KD()
        self.solve_b()
        self.solve_remaining()

    def generate_b(self):
        phi_b = np.zeros((self.nb, self.nb))
        for i, p in enumerate(self.base_set):
            for j, q in enumerate(self.base_set):
                r = np.linalg.norm(self.X[p] - self.X[q]) / self.radii[q]
                if r <= 1.0:
                    phi_b[i, j] = c2(r)

        # fill via symmetry
        for i in range(self.nb):
            for j in range(self.nb):
                coef = max([phi_b[i, j], phi_b[j, i]])
                phi_b[i, j] = coef
                phi_b[j, i] = coef

        self.phi_b = phi_b

    def generate_b_KD(self):
        phi_b = np.zeros((self.nb, self.nb))
        X_base = self.X[0:self.nb, :]
        X_tree = KDTree(X_base, leaf_size=10)

        for i, p in enumerate(self.base_set):
            ind, dist = X_tree.query_radius(
                [self.X[p, :]], self.r, return_distance=True)
            for index, rad in zip(ind[0], dist[0]):
                phi_b[i, index] = c2(rad / self.r)

        # fill via symmetry
        for i in range(self.nb):
            for j in range(self.nb):
                coef = max([phi_b[i, j], phi_b[j, i]])
                phi_b[i, j] = coef
                phi_b[j, i] = coef

        self.phi_b = phi_b

    def generate_full_mat(self):
        phi_b = np.zeros((self.np, self.np))
        for i in range(self.np):
            for j in range(self.np):
                r = np.linalg.norm(self.X[i, :] - self.X[j, :]) / self.r
                if r <= 1.0:
                    phi_b[i, j] = c2(r)

        # fill via symmetry
        for i in range(self.np):
            for j in range(self.np):
                coef = max([phi_b[i, j], phi_b[j, i]])
                phi_b[i, j] = coef
                phi_b[j, i] = coef

        self.phi_b = phi_b

    def generate_r(self):
        phi_r = np.zeros((self.np - self.nb, self.nb))
        for i, p in enumerate(self.remaining_set):
            for j, q in enumerate(self.base_set):
                r = np.linalg.norm(self.X[p] - self.X[q]) / self.radii[q]
                if r <= 1.0:
                    phi_r[i, j] = c2(r)

        self.phi_r = phi_r

    def generate_r_KD(self):
        phi_r = np.zeros((self.np - self.nb, self.nb))
        X_remaining = self.X[self.nb:, :]
        X_tree = KDTree(X_remaining, leaf_size=10)

        for i, p in enumerate(self.base_set):
            ind, dist = X_tree.query_radius(
                [self.X[p, :]], self.radii[p], return_distance=True)

            for index, rad in zip(ind[0], dist[0]):
                phi_r[index, i] = c2(rad / self.radii[i])

        self.phi_r = phi_r

    def generate_LCRS(self):
        LCRS = np.zeros((self.np - self.nb, self.np - self.nb))
        for i, p in enumerate(self.remaining_set):
            for j in range(i+1):
                q = self.remaining_set[j]
                r = np.linalg.norm(self.X[p] - self.X[q]) / self.radii[q]
                if r <= 1.0:
                    LCRS[i, j] = c2(r)

        self.LCRS = sparse.csc_matrix(LCRS)

    def generate_LCRS_KD(self):
        LCRS = np.zeros((self.np - self.nb, self.np - self.nb))
        X_remaining = self.X[self.nb:, :]
        X_tree = KDTree(X_remaining, leaf_size=10)
        for i, p in enumerate(self.remaining_set):
            ind, dist = X_tree.query_radius(
                [self.X[p, :]], self.radii[p], return_distance=True)

            for index, rad in zip(ind[0], dist[0]):
                LCRS[i, index] = c2(rad / self.radii[p])

        self.LCRS = sparse.csc_matrix(LCRS)

    def solve_b(self):
        base_dX = self.dX[self.base_set, :].copy()
        lu, piv = lu_factor(self.phi_b)
        base_coef = lu_solve((lu, piv), base_dX)

        self.coef = np.zeros_like(dX)
        self.coef[:self.nb, :] = base_coef

    def solve_remaining(self):
        dX_res = self.dX.copy()
        # update residual
        dX_res[self.nb:, :] = dX_res[self.nb:, :] - \
            self.phi_r @ self.coef[:self.nb, :]

        for i, p in enumerate(self.remaining_set):
            self.coef[i + self.nb, :] = dX_res[i + self.nb, :]
            for j in range(self.LCRS.indptr[i], self.LCRS.indptr[i+1]):
                ptr = self.LCRS.indices[j]
                coef = self.LCRS.data[j]
                dX_res[ptr + self.nb, :] = dX_res[ptr + self.nb, :] - \
                    coef * self.coef[i + self.nb, :]

    def preprov_V(self, V):
        self.V = V.copy()
        self.nV = V.shape[0]

        col_index_temp = []
        psi_v_rowptr = np.zeros(self.nV + 1, dtype=int)

        j = 0

        for i in range(self.nV):
            col_index_temp.append(0)
            j += 1
            for q in self.remaining_set:
                r = np.linalg.norm(self.V[i, :] - self.X[q, :])
                e = r / self.radii[q]
                if e < 1:
                    col_index_temp.append(int(q))
                    j += 1
            psi_v_rowptr[i + 1] = int(j)

        # self.psi_v_rowptr = psi_v_rowptr
        # self.col_index = col_index_temp

        print('done')

    def preprov_V_KD(self, V):
        self.V = V.copy()
        self.nV = V.shape[0]

        X_tree = KDTree(self.V, leaf_size=10, metric='euclidean')

        psi_v = [[0] for i in range(self.nV)]
        psi_v_val_temp = [[0] for i in range(self.nV)]

        for i in self.remaining_set:
            ind, dist = X_tree.query_radius(
                [self.X[i]], self.radii[i], return_distance=True)
            for index, rad in zip(ind[0], dist[0]):
                psi_v[index].append(int(i))
                psi_v_val_temp[index].append(c2(rad / self.radii[i]))
                if rad / self.radii[i] > 1:
                    print("ERROR")

        psi_v_rowptr = [0]
        col_index_temp = []
        psi_v_val = []

        k = 0

        for i in range(self.nV):
            k += len(psi_v[i])
            psi_v_rowptr.append(k)
            psi_v_val.append(0)
            for j in range(len(psi_v[i])):
                col_index_temp.append(psi_v[i][j])
                psi_v_val.append(psi_v_val_temp[i][j])

        self.psi_v_rowptr = psi_v_rowptr
        self.col_index = col_index_temp
        self.psi_v_val = psi_v_val

    def transfer_KD(self):
        self.dV = np.zeros_like(self.V)
        n_b_coef = 0
        n_q_coef = 0
        n_coef = 0
        for i in range(self.nV):
            for k in range(self.psi_v_rowptr[i], self.psi_v_rowptr[i + 1]):
                if self.col_index[k] == 0:
                    for q in self.base_set:
                        r = np.linalg.norm(self.V[i, :] - self.X[q, :])
                        e = r / self.radii[q]

                        if e <= 1:
                            coef = c2(e)
                            self.dV[i, :] += coef * self.coef[q, :]
                            n_b_coef += 1
                            n_coef += 1
                else:
                    q = self.col_index[k]
                    # r = np.linalg.norm(self.V[i, :] - self.X[q, :])
                    # e = r / self.radii[q]

                    # if e <= 1:
                    #     coef = c2(e)
                    self.dV[i, :] += self.psi_v_val[k] * self.coef[q, :]
                    n_q_coef += 1
                    n_coef += 1

        self.n_b_coef = n_b_coef
        self.n_q_coef = n_q_coef
        self.n_coef = n_coef

    def transfer(self):
        self.dV = np.zeros_like(self.V)
        n_b_coef = 0
        n_q_coef = 0
        n_coef = 0
        for i in range(self.nV):
            for k in range(self.psi_v_rowptr[i], self.psi_v_rowptr[i + 1]):
                if self.col_index[k] == 0:
                    for q in self.base_set:
                        r = np.linalg.norm(self.V[i, :] - self.X[q, :])
                        e = r / self.radii[q]

                        if e <= 1:
                            coef = c2(e)
                            self.dV[i, :] += coef * self.coef[q, :]
                            n_b_coef += 1
                            n_coef += 1
                else:
                    q = self.col_index[k]
                    r = np.linalg.norm(self.V[i, :] - self.X[q, :])
                    e = r / self.radii[q]

                    if e <= 1:
                        coef = c2(e)
                    self.dV[i, :] += coef * self.coef[q, :]
                    n_q_coef += 1
                    n_coef += 1

        self.n_b_coef = n_b_coef
        self.n_q_coef = n_q_coef
        self.n_coef = n_coef

    def reorder(self):
        X_new = self.X[self.active_list, :]
        dX_new = self.dX[self.active_list, :]
        radii_new = [self.radii[i] for i in self.active_list]

        self.X = X_new
        self.dX = dX_new
        self.radii = radii_new
        self.base_set = [i for i in range(self.nb)]
        self.remaining_set = [i for i in range(self.nb, self.np)]


def c2(r):
    psi = ((1 - r)**4) * (4 * r + 1)
    return psi


if __name__ == "__main__":
    start = time.time()

    X = np.loadtxt(
        '/home/tom/Documents/University/Coding/zCFD_Utils/data/surface.xyz', skiprows=1)
    V = np.loadtxt(
        '/home/tom/Documents/University/Coding/zCFD_Utils/data/volume.xyz', skiprows=1)
    dX = np.loadtxt(
        '/home/tom/Documents/University/Coding/zCFD_Utils/data/displacements.xyz', skiprows=1)

    nb = 0.1
    r = 4
    t = np.deg2rad(10)

    rot_vector = np.array(
        [[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])

    dX = X @ rot_vector - X

    M = MultiScale(X, nb, r)

    start = time.time()

    M.sequence_control_points(parent_weighting=True)

    end = time.time()

    print('Total time: ')
    print(end - start)

    # M2 = MultiScale(X, nb, r)
    # M2.sequence_control_points(parent_weighting=False)

    # print("done")

    M.multiscale_solve(dX)

    M.preprov_V_KD(V)
    M.transfer()

    plt.plot(X[:, 0], X[:, 1])
    plt.plot(X[:, 0] + dX[:, 0], X[:, 1] + dX[:, 1])

    plt.plot(V[:, 0], V[:, 1])
    plt.plot(V[:, 0] + M.dV[:, 0], V[:, 1] + M.dV[:, 1])

    # plt.plot(OD[:, 0], OD[:, 1])

    plt.show()
