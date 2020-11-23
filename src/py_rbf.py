import numpy as np

# Functions


def generate_transfer_matrix(aero_nodes,struct_nodes,r0,polynomial=True):
    # returns- H: (n_a,n_s) full transfer matrix between STRUCTURAL NODES and AERODYNAMIC NODES
    n_a = len(aero_nodes[:, 0])
    n_s = len(struct_nodes[:, 0])

    if polynomial:
        A_as = np.zeros((n_a, n_s+4))
        P_s = np.ones((4, n_s))
    else:
        A_as = np.zeros((n_a, n_s))

    M_ss = np.zeros((n_s, n_s))

    for i in range(n_s):
        for j in range(n_s):
            rad = (np.linalg.norm((struct_nodes[i] - struct_nodes[j])))/r0
            if rad <= 1.0:
                M_ss[i][j] = ((1-rad)**4) * (4*rad+1)  # Wendland C2
        if polynomial:
            P_s[1:, i] = struct_nodes[i]

    for i in range(n_a):
        for j in range(n_s):
            rad = np.linalg.norm((aero_nodes[i]-struct_nodes[j]))/r0
            if rad <= 1.0:
                if polynomial:
                    A_as[i][j+4] = ((1-rad)**4) * (4*rad+1)  # Wendland C2
                else:
                    A_as[i][j] = ((1-rad)**4) * (4*rad+1)  # Wendland C2
        if polynomial:
            A_as[i][1:4] = aero_nodes[i]
            A_as[i][0] = 1

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
        F_s[:, i] = np.matmul(np.transpose(H), F_s[:, i])
    return F_s
