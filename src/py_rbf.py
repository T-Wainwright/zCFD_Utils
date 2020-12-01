import numpy as np

# Functions


def generate_transfer_matrix(aero_nodes, struct_nodes, r0, rbf='c2', polynomial=True):
    # returns- H: (n_a,n_s) full transfer matrix between STRUCTURAL NODES and AERODYNAMIC NODES
    n_a = len(aero_nodes[:, 0])
    n_s = len(struct_nodes[:, 0])   

    switcher = {'c0': c0, 'c2': c2, 'c4': c4, 'c6': c6}
    rbf = switcher.get(rbf)

    # preallocate matrices
    if polynomial:
        A_as = np.zeros((n_a, n_s + 4))
        P_s = np.ones((4, n_s))
    else:
        A_as = np.zeros((n_a, n_s))

    M_ss = np.zeros((n_s, n_s))

    # Generate block matrix M (and P if polynomial) equations 11 and 12
    for i in range(n_s):
        for j in range(n_s):
            rad = (np.linalg.norm((struct_nodes[i] - struct_nodes[j]))) / r0
            if rad <= 1.0:
                M_ss[i][j] = rbf(rad)
        if polynomial:
            P_s[1:, i] = struct_nodes[i]

    # Generate A_as matrix- equation 13
    for i in range(n_a):
        for j in range(n_s):
            rad = np.linalg.norm((aero_nodes[i] - struct_nodes[j])) / r0
            if rad <= 1.0:
                if polynomial:
                    A_as[i][j + 4] = rbf(rad)
                else:
                    A_as[i][j] = rbf(rad)
        if polynomial:
            A_as[i][1:4] = aero_nodes[i]
            A_as[i][0] = 1

    np.savetxt("../data/M_ss.csv", M_ss, delimiter=",")
    # np.savetxt("../data/P_s.csv", P_s, delimiter=",")
    np.savetxt("../data/A_as.csv", A_as, delimiter=",")

    # Invert M matrix- Moore-Penrose inversion
    M_inv = np.linalg.pinv(M_ss)

    if polynomial:
        # Equations 21 and 22
        M_p = np.linalg.pinv(np.matmul(np.matmul(P_s, M_inv), np.transpose(P_s)))

        Top = np.matmul(np.matmul(M_p, P_s), M_inv)
        Bottom = M_inv - np.matmul(np.matmul(np.matmul(np.matmul(M_inv, np.transpose(P_s)), M_p), P_s), M_inv)

        B = np.concatenate((Top, Bottom))

        H = np.matmul(A_as, B)


        L = np.concatenate((np.zeros((4,4)),np.transpose(P_s)))
        R = np.concatenate((P_s,M_ss))
        Css = np.concatenate((L,R),axis=1)
        H = np.matmul(A_as, np.linalg.pinv(Css))
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
