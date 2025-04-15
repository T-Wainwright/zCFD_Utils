import numpy as np
from scipy import sparse
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import time
import zcfdutils.py_rbf
from matplotlib import rc, rcParams, cycler


class Multiscale:
    """Multiscale interpolator which uses non accelerated point selection algorithm"""

    def __init__(
        self,
        source_points,
        base_point_fraction,
        base_support_radius,
        rbf="c2",
        incLinearPolynomial=False,
        parent_weighting=False,
        useKDTrees=True,
    ) -> None:
        self.source_points = source_points
        self.n_control_points = self.source_points.shape[0]
        self.n_base_set = int(np.ceil(base_point_fraction * self.n_control_points))
        self.base_support_radius = base_support_radius
        self.poly = incLinearPolynomial
        self.parent_weighting = parent_weighting
        self.KD = useKDTrees
        self.nops = {}

        switcher = {"c0": self.c0, "c2": self.c2, "c4": self.c4, "c6": self.c6}
        self.rbf = switcher.get(rbf)

        self.sequence_control_points()

    # Main functional method calls

    def sequence_control_points(self):
        # Create active lists
        active_list = []
        inactive_list = [i for i in range(self.n_control_points)]
        sep_dist = [1e10 for i in range(self.n_control_points)]
        radii = [self.base_support_radius for i in range(self.n_control_points)]
        parent = [0 for i in range(self.n_control_points)]
        base_set = []
        remaining_set = []

        n_active = 0

        # Grab first control point
        active_node = inactive_list[0]
        active_list.append(active_node)
        inactive_list.remove(active_node)
        base_set.append(active_node)

        n_active += 1

        # build KD Tree to speed up radius searches

        X_tree = KDTree(self.source_points, leaf_size=10, metric="euclidean")

        # Cycle over remaining points

        nop = 0

        while n_active < self.n_control_points:
            if self.KD:
                ind, dist = X_tree.query_radius(
                    [self.source_points[active_node]],
                    sep_dist[active_node],
                    return_distance=True,
                )
            else:
                dist, ind = X_tree.query(
                    [self.source_points[active_node]], self.n_control_points
                )

            nop += 1

            sep_dist[active_node] = -1e-10

            for i, p in enumerate(ind[0]):
                nop += 1
                if dist[0][i] < sep_dist[p]:
                    sep_dist[p] = dist[0][i]
                    parent[p] = active_node

            sep_dist_temp = sep_dist.copy()

            if self.parent_weighting:
                # if searching base set, find the node with the max separation from the last added node
                if n_active < self.n_base_set:
                    nop += 1
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

            if n_active < self.n_base_set:
                radii[active_node] = self.base_support_radius
                base_set.append(active_node)
            else:
                radii[active_node] = sep_dist[active_node]
                remaining_set.append(active_node)

            n_active += 1

        self.active_list = active_list
        self.radii = radii
        self.base_set = base_set
        self.remaining_set = remaining_set

        self.nops["multiscale refinement"] = nop

    def preprocess_target_mesh(self, target_mesh):
        if self.KD:
            self.preprov_V_KD(target_mesh)
        else:
            self.preprov_V(target_mesh)

    def multiscale_solve(self, source_data):
        self.source_data = source_data.copy()
        self.reorder()

        if self.KD:
            # KD Tree optimisation
            self.generate_b_KD()
            self.generate_r_KD()
            self.generate_LCRS_KD()
        else:
            self.generate_b()
            self.generate_r()
            self.generate_LCRS()

        if self.poly:
            # Generate and solve polynomial
            self.generate_P()
            self.solve_a()

        self.solve_b()
        self.solve_remaining()

    def transfer_solution(self):
        self.dV_rbf = np.zeros_like(self.target_mesh)
        n_oper_base = 0
        n_oper_refine = 0

        for i in range(self.n_target_points):
            for k in range(self.psi_v.indptr[i], self.psi_v.indptr[i + 1]):
                if self.psi_v.indices[k] == 0:
                    for q in self.base_set:
                        r = np.linalg.norm(
                            self.target_mesh[i, :] - self.source_points[q, :]
                        )
                        e = r / self.radii[q]

                        if e <= 1:
                            coef = self.rbf(e)
                            self.dV_rbf[i, :] += coef * self.coef[q, :]
                            coef1 = coef
                            n_oper_base += 1

                else:
                    q = self.psi_v.indices[k]
                    r = np.linalg.norm(
                        self.target_mesh[i, :] - self.source_points[q, :]
                    )
                    e = r / self.radii[q]

                    if e <= 1:
                        coef = self.rbf(e)
                        coef2 = coef
                        self.dV_rbf[i, :] += coef * self.coef[q, :]
                        n_oper_refine += 1

        if self.poly:
            self.dV_poly = np.zeros_like(self.target_mesh)
            self.dV_poly = self.A_poly @ self.a_poly
            self.dV = self.dV_rbf + self.dV_poly
        else:
            self.dV = self.dV_rbf

        print(n_oper_base)
        print(n_oper_refine)

    # Volume mesh preprocessing calls

    def preprov_V(self, V):
        self.target_mesh = V.copy()
        self.n_target_points = V.shape[0]

        nop = 0

        col_index_temp = []
        psi_v_rowptr = np.zeros(self.n_target_points + 1, dtype=int)

        j = 0

        for i in range(self.n_target_points):
            col_index_temp.append(0)
            j += 1
            for q in self.remaining_set:
                nop += 1
                r = np.linalg.norm(self.target_mesh[i, :] - self.source_points[q, :])
                e = r / self.radii[q]
                if e < 1:
                    col_index_temp.append(int(q))
                    j += 1
            psi_v_rowptr[i + 1] = int(j)

        data = [0 for i in col_index_temp]

        self.psi_v = sparse.csc_matrix((data, col_index_temp, psi_v_rowptr))

        self.nops["preprocess v"] = nop

    def preprov_V_KD(self, target_mesh):
        self.target_mesh = target_mesh.copy()
        self.n_target_points = target_mesh.shape[0]
        nop = 0

        X_tree = KDTree(self.target_mesh, leaf_size=10, metric="euclidean")
        nop += 1

        data = []
        row_index = []
        col_index = []

        for i in range(self.n_control_points):
            ind, dist = X_tree.query_radius(
                [self.source_points[i]], self.radii[i], return_distance=True
            )
            nop += 1
            for index, rad in zip(ind[0], dist[0]):
                nop += 1
                row_index.append(index)
                col_index.append(i)
                data.append(self.rbf(rad / self.radii[i]))
                if rad / self.radii[i] > 1:
                    print("ERROR")

        psi_v_rowptr = [0]
        col_index_temp = []
        psi_v_val = []

        k = 0

        self.psi_v = sparse.csr_matrix(
            (data, (row_index, col_index)),
            shape=(self.n_target_points, self.n_control_points),
        )

        if self.poly:
            self.A_poly = np.ones((self.n_target_points, 4))
            self.A_poly[:, 1:4] = self.target_mesh

        self.nops["preprocess v"] = nop

    # Matrix Generation

    def generate_P(self):
        self.P = np.ones((4, self.n_control_points))
        self.P[1:4, :] = self.source_points.T

        print("done")

    def generate_P_reverse(self):
        self.P_reverse = np.ones((4, self.n_target_points))
        self.P_reverse[1:4, :] = self.target_mesh.T

    def generate_b(self):
        nop = 0
        phi_b = np.zeros((self.n_base_set, self.n_base_set))
        for i, p in enumerate(self.base_set):
            for j, q in enumerate(self.base_set):
                nop += 1
                r = (
                    np.linalg.norm(self.source_points[p] - self.source_points[q])
                    / self.radii[q]
                )
                if r <= 1.0:
                    phi_b[i, j] = self.rbf(r)

        # fill via symmetry
        for i in range(self.n_base_set):
            for j in range(self.n_base_set):
                nop += 1
                coef = max([phi_b[i, j], phi_b[j, i]])
                phi_b[i, j] = coef
                phi_b[j, i] = coef

        self.phi_b = phi_b
        self.nops["generate b"] = nop

    def generate_b_KD(self):
        nop = 0
        phi_b = np.zeros((self.n_base_set, self.n_base_set))
        X_base = self.source_points[0 : self.n_base_set, :]
        X_tree = KDTree(X_base, leaf_size=10)
        nop += 1

        for i, p in enumerate(self.base_set):
            ind, dist = X_tree.query_radius(
                [self.source_points[p, :]],
                self.base_support_radius,
                return_distance=True,
            )
            nop += 1
            for index, rad in zip(ind[0], dist[0]):
                nop += 1
                phi_b[i, index] = self.rbf(rad / self.base_support_radius)

        # fill via symmetry
        for i in range(self.n_base_set):
            for j in range(self.n_base_set):
                nop += 1
                coef = max([phi_b[i, j], phi_b[j, i]])
                phi_b[i, j] = coef
                phi_b[j, i] = coef

        self.phi_b = phi_b
        self.nops["generate b"] = nop

    def generate_full_mat(self):
        phi_b = np.zeros((self.n_control_points, self.n_control_points))
        for i in range(self.n_control_points):
            for j in range(self.n_control_points):
                r = (
                    np.linalg.norm(self.source_points[i, :] - self.source_points[j, :])
                    / self.base_support_radius
                )
                if r <= 1.0:
                    phi_b[i, j] = self.rbf(r)

        # fill via symmetry
        for i in range(self.n_control_points):
            for j in range(self.n_control_points):
                coef = max([phi_b[i, j], phi_b[j, i]])
                phi_b[i, j] = coef
                phi_b[j, i] = coef

        self.phi_b = phi_b

    def generate_r(self):
        nop = 0
        phi_r = np.zeros((self.n_control_points - self.n_base_set, self.n_base_set))
        for i, p in enumerate(self.remaining_set):
            for j, q in enumerate(self.base_set):
                nop += 1
                r = (
                    np.linalg.norm(self.source_points[p] - self.source_points[q])
                    / self.radii[q]
                )
                if r <= 1.0:
                    phi_r[i, j] = self.rbf(r)

        self.phi_r = phi_r
        self.nops["generate r"] = nop

    def generate_r_KD(self):
        nop = 1
        phi_r = np.zeros((self.n_control_points - self.n_base_set, self.n_base_set))
        X_remaining = self.source_points[self.n_base_set :, :]
        X_tree = KDTree(X_remaining, leaf_size=10)
        nop += 1

        for i, p in enumerate(self.base_set):
            ind, dist = X_tree.query_radius(
                [self.source_points[p, :]], self.radii[p], return_distance=True
            )
            nop += 1
            for index, rad in zip(ind[0], dist[0]):
                nop += 1
                phi_r[index, i] = self.rbf(rad / self.radii[i])

        self.phi_r = phi_r
        self.nops["generate r"] = nop

    def generate_LCRS(self):
        nop = 0
        LCRS = np.zeros(
            (
                self.n_control_points - self.n_base_set,
                self.n_control_points - self.n_base_set,
            )
        )
        for i, p in enumerate(self.remaining_set):
            for j in range(i + 1):
                nop += 1
                q = self.remaining_set[j]
                r = (
                    np.linalg.norm(self.source_points[p] - self.source_points[q])
                    / self.radii[q]
                )
                if r <= 1.0:
                    LCRS[i, j] = self.rbf(r)

        self.LCRS = sparse.csc_matrix(LCRS)
        self.nops["generate LCRS"] = nop

    def generate_LCRS_KD(self):
        nop = 0
        LCRS = np.zeros(
            (
                self.n_control_points - self.n_base_set,
                self.n_control_points - self.n_base_set,
            )
        )
        X_remaining = self.source_points[self.n_base_set :, :]
        X_tree = KDTree(X_remaining, leaf_size=10)
        nop += 1
        for i, p in enumerate(self.remaining_set):
            ind, dist = X_tree.query_radius(
                [self.source_points[p, :]], self.radii[p], return_distance=True
            )
            nop += 1

            for index, rad in zip(ind[0], dist[0]):
                nop += 1
                LCRS[i, index] = self.rbf(rad / self.radii[p])

        self.LCRS = sparse.csc_matrix(LCRS)
        self.nops["generate LCRS"] = nop

    # Solve calls

    def solve_a(self):
        self.a_poly = np.linalg.pinv(self.P).T @ self.source_data

        print("done")

    def solve_b(self):
        if self.poly:
            self.rhs = (
                self.source_data
                - self.P.T
                @ np.linalg.pinv(self.P @ self.P.T)
                @ self.P
                @ self.source_data
            )
        else:
            self.rhs = self.source_data
        base_rhs = self.rhs[self.base_set, :].copy()
        lu, piv = lu_factor(self.phi_b)
        base_coef = lu_solve((lu, piv), base_rhs)

        self.coef = np.zeros_like(self.source_data)
        self.coef[: self.n_base_set, :] = base_coef

    def solve_remaining(self):
        dX_res = self.rhs.copy()
        # update residual
        dX_res[self.n_base_set :, :] = (
            dX_res[self.n_base_set :, :] - self.phi_r @ self.coef[: self.n_base_set, :]
        )

        for i, p in enumerate(self.remaining_set):
            self.coef[i + self.n_base_set, :] = dX_res[i + self.n_base_set, :]
            for j in range(self.LCRS.indptr[i], self.LCRS.indptr[i + 1]):
                ptr = self.LCRS.indices[j]
                coef = self.LCRS.data[j]
                dX_res[ptr + self.n_base_set, :] = (
                    dX_res[ptr + self.n_base_set, :]
                    - coef * self.coef[i + self.n_base_set, :]
                )

    def reorder(self):
        X_new = self.source_points[self.active_list, :]
        dX_new = self.source_data[self.active_list, :]
        radii_new = [self.radii[i] for i in self.active_list]

        self.source_points = X_new
        self.source_data = dX_new
        self.radii = radii_new
        self.base_set = [i for i in range(self.n_base_set)]
        self.remaining_set = [i for i in range(self.n_base_set, self.n_control_points)]

    def solve_reverse_poly(self, dV):
        return np.linalg.pinv(self.P_reverse).T @ dV

    def generate_A_poly_reverse(self):
        self.A_poly_reverse = np.ones((4, self.n_control_points))
        self.A_poly_reverse[1:4, :] = self.source_points.T

    def reverse_interpolate(self, dV):
        self.reverse_coefficients = np.zeros_like(self.target_mesh)
        if self.poly:
            self.generate_P_reverse()
            self.generate_A_poly_reverse()
            a_poly_reverse = self.solve_reverse_poly(dV)
            self.dX_i_poly = self.A_poly_reverse.T @ a_poly_reverse
            self.rhs_reverse = (
                dV
                - self.P_reverse.T
                @ np.linalg.pinv(self.P_reverse @ self.P_reverse.T)
                @ self.P_reverse
                @ dV
            )
        else:
            self.rhs_reverse = dV

        for i in range(self.n_target_points):
            for k in range(self.psi_v.indptr[i], self.psi_v.indptr[i + 1]):
                if self.psi_v.indices[k] == 0:
                    for q in self.base_set:
                        r = np.linalg.norm(
                            self.target_mesh[i, :] - self.source_points[q, :]
                        )
                        e = r / self.radii[q]

                        if e <= 1:
                            coef = self.rbf(e)
                            self.reverse_coefficients[i, :] += (
                                coef * self.rhs_reverse[i, :]
                            )

                else:
                    q = self.psi_v.indices[k]
                    r = np.linalg.norm(
                        self.target_mesh[i, :] - self.source_points[q, :]
                    )
                    e = r / self.radii[q]

                    if e <= 1:
                        coef = self.rbf(e)
                        self.reverse_coefficients[i, :] += coef * self.rhs_reverse[i, :]

    def reverse_solve(self):
        reverse_coefficients_residual = self.reverse_coefficients.copy()
        self.dX_i_rbf = np.zeros_like(self.source_points)
        for i in range(self.n_control_points - 1, self.n_base_set, -1):
            self.dX_i_rbf[i, :] = self.reverse_coefficients[i, :]
            for j, p in enumerate(self.LCRS.indices):
                r = (
                    min(
                        np.argmin(
                            abs(self.LCRS.indptr - j * np.ones_like(self.LCRS.indptr))
                        ),
                        np.argmin(
                            abs(j * np.ones_like(self.LCRS.indptr) - self.LCRS.indptr)
                        ),
                    )
                    + self.n_base_set
                )
                if p == 1:
                    coef = self.LCRS.data[j]
                    reverse_coefficients_residual[r, :] -= coef * self.dX_i_rbf[i, :]

        reverse_coefficients_residual[0 : self.n_base_set, :] -= (
            self.phi_r.T @ self.dX_i_rbf[self.n_base_set :, :]
        )

        lu, piv = lu_factor(self.phi_b)
        self.dX_i_rbf[0 : self.n_base_set] = lu_solve(
            (lu, piv), reverse_coefficients_residual[0 : self.n_base_set, :]
        )

        if self.poly:
            self.dX_i = self.dX_i_rbf + self.dX_i_poly
        else:
            self.dX_i = self.dX_i_rbf

    def print_total_nops(self):
        nop = 0
        for _, i in self.nops.items():
            nop += i
        print("Total nops = {}".format(nop))

    @staticmethod
    def c0(r):
        psi = (1 - r) ** 2
        return psi

    @staticmethod
    def c2(r):
        psi = ((1 - r) ** 4) * (4 * r + 1)
        return psi

    @staticmethod
    def c4(r):
        psi = ((1 - r) ** 6) * (35 * r**2 + 18 * r + 3)
        return psi

    @staticmethod
    def c6(r):
        psi = ((1 - r) ** 8) * (32 * r**3 + 25 * r**2 + 8 * r + 1)
        return psi


# if __name__ == "__main__":
# start = time.time()

# X = np.loadtxt(
#     '/Users/tom.wainwright/Documents/code/zCFD_Utils/data/surface.xyz', skiprows=1)
# V = np.loadtxt(
#     '/Users/tom.wainwright/Documents/code/zCFD_Utils/data/volume.xyz', skiprows=1)
# dX = np.loadtxt(
#     '/Users/tom.wainwright/Documents/code/zCFD_Utils/data/displacements.xyz', skiprows=1)

# a = np.linspace(0, 2 * np.pi, 100)

# # X behaves as a spar
# structural_surface = np.zeros((100, 3))
# structural_surface[:, 0] = 0.01 * np.cos(a) + 0.25
# structural_surface[:, 1] = 0.01 * np.sin(a)

# # dX is displacement of spar
# structural_displacement = np.zeros_like(structural_surface)
# structural_displacement[:, 0] = 0.0
# structural_displacement[:, 1] = 1.0

# # dV_i is force on aerodynamic surface- in this case constant X forcing
# aero_forces = np.zeros_like(X)
# aero_forces[:, 0] = 2.0
# aero_forces[:, 1] = 1.0

# nb = 0.1
# r = 100
# t = np.deg2rad(10)

# rot_vector = np.array(
#     [[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])

# dX = X @ rot_vector - X

# scales = 8

# print(X.shape)

# refinemement_ops_old = []
# pts = []

# for i in range(scales):
#     M = LegacyMultiScale(X, nb, r, incLinearPolynomial=False, useKDTrees=False)
#     # M.preprocess_target_mesh(V)
#     # M.multiscale_solve(dX)
#     # M.transfer_solution()

#     M.print_total_nops()
#     refinemement_ops_old.append(M.nops["multiscale refinement"])
#     pts.append(X.shape[0])

#     # double size of X
#     X_new = np.concatenate((X, X), axis=0)
#     dX_new = np.concatenate((dX, dX), axis=0)

#     print(X_new.shape)
#     X = X_new
#     dX = dX_new

# refinemement_ops_new = []
# X = np.loadtxt(
#     '/Users/tom.wainwright/Documents/code/zCFD_Utils/data/surface.xyz', skiprows=1)

# for i in range(scales):
#     M = LegacyMultiScale(X, nb, r, incLinearPolynomial=False, useKDTrees=True)
#     # M.preprocess_target_mesh(V)
#     # M.multiscale_solve(dX)
#     # M.transfer_solution()

#     M.print_total_nops()
#     refinemement_ops_new.append(M.nops["multiscale refinement"])

#     # double size of X
#     X_new = np.concatenate((X, X), axis=0)
#     dX_new = np.concatenate((dX, dX), axis=0)

#     print(X_new.shape)
#     X = X_new
#     dX = dX_new


# # plt.plot(X[:, 0], X[:, 1])
# # plt.plot(X[:, 0] + dX[:, 0],
# #          X[:, 1] + dX[:, 1])

# # plt.plot(X[:, 0], X[:, 1])
# # plt.plot(X[:, 0] + dX[:, 0],
# #          X[:, 1] + dX[:, 1])

# print(M.nops)

# # plt.plot(V[:, 0], V[:, 1])
# # plt.plot(V[:, 0] + M.dV_poly[:, 0], V[:, 1] + M.dV_poly[:, 1])

# # plt.plot(OD[:, 0], OD[:, 1])

# diff = [i/j for i, j in zip(refinemement_ops_old, refinemement_ops_new)]

# # plt.semilogy(pts, refinemement_ops_old)
# # plt.semilogy(pts, refinemement_ops_new)


# rcParams["axes.prop_cycle"] = cycler(
#     "color",
#     [
#         "k",
#         "#1f77b4",
#         "#ff7f0e",
#         "#2ca02c",
#         "#d62728",
#         "#9467bd",
#         "#8c564b",
#         "#e377c2",
#         "#7f7f7f",
#         "#bcbd22",
#         "#17becf",
#     ],
# )

# fig1 = plt.figure(figsize=(6, 5), dpi=100, facecolor="w", edgecolor="k")
# ax1 = fig1.gca()

# fig2 = plt.figure(figsize=(6, 5), dpi=100, facecolor="w", edgecolor="k")
# ax2 = fig2.gca()

# ax1.plot(pts, diff, '-d')
# ax1.set_xlabel("Number of Sample Points [-]", fontsize=20, fontweight="bold")
# ax1.set_ylabel("Acceleration Factor [-]", fontsize=20, fontweight="bold")
# ax1.tick_params(labelsize=20)
# ax1.grid(True)
# # ax1.set_yscale("log")
# # ax1.set_xscale("log")

# ax2.plot(pts, refinemement_ops_old, '-d', label="Original Multiscale")
# ax2.plot(pts, refinemement_ops_new, '-s', label="Fast Multiscale")
# ax2.set_yscale("log")
# ax2.set_xscale("log")
# ax2.set_xlabel("Number of Sample Points [-]", fontsize=20, fontweight="bold")
# ax2.set_ylabel("Number of Operations [-]", fontsize=20, fontweight="bold")
# ax2.tick_params(labelsize=20)
# ax2.legend(fontsize=15)
# ax2.grid(True)

# fig1.tight_layout()
# fig2.tight_layout()

# fig1.savefig("/Users/tom.wainwright/Documents/code/zCFD_Utils/data/multiAcceleration.png", format="png")
# fig1.savefig("/Users/tom.wainwright/Documents/code/zCFD_Utils/data/multiAcceleration.eps", format="eps")

# fig2.savefig("/Users/tom.wainwright/Documents/code/zCFD_Utils/data/multiScaling.png", format="png")
# fig2.savefig("/Users/tom.wainwright/Documents/code/zCFD_Utils/data/multiScaling.eps", format="eps")


# plt.show()


def PointsInCircum(r, n=100):
    return np.array(
        [
            (np.cos(2 * np.pi / n * x) * r, np.sin(2 * np.pi / n * x) * r)
            for x in range(0, n + 1)
        ]
    )


def squarePoints(l):
    return 0.5 * np.array(
        [[-l, -l], [-l, 0], [-l, l], [0, l], [l, l], [l, 0], [l, -l], [0, -l]]
    )


def loopvec(v):
    return np.append(v, v[0])


if __name__ == "__main__":

    start = time.time()

    eval = PointsInCircum(1, 20)
    cont = squarePoints(1.0)

    disp = 0.0 * -np.ones(cont.shape)
    disp[4, 0] -= 0.25
    disp[4, 1] -= 0.25

    r0 = 1.0
    nb = 1.0
    # def __init__(self, source_points, base_point_fraction, base_support_radius, rbf='c2', incLinearPolynomial=False, parent_weighting=False, useKDTrees=True) -> None:

    M = MultiScale(cont, nb, r0, incLinearPolynomial=False, parent_weighting=False)
    M.preprocess_target_mesh(eval)
    M.multiscale_solve(disp)
    M.transfer_solution()
    deval = M.dV
    print(deval)

    plt.plot(loopvec(eval[:, 0]), loopvec(eval[:, 1]))
    plt.plot(loopvec(cont[:, 0]), loopvec(cont[:, 1]))
    eval2 = eval + deval
    cont2 = cont + disp
    plt.plot(loopvec(eval2[:, 0]), loopvec(eval2[:, 1]))
    plt.plot(loopvec(cont2[:, 0]), loopvec(cont2[:, 1]))
    plt.title("Multiscale ")
    plt.show()

    # dshape = deval.shape
    # deval2 = np.ones(dshape)
    # M.reverse_interpolate(deval2)
    # M.reverse_solve()
    # print(M.dX_i)
