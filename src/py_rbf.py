import numpy as np

# Functions

class UoB_coupling():

    def __init__(self):
        pass

    def load_struct(self, fname):
        self.struct_nodes = np.loadtxt(fname, skiprows=1)
        self.n_s = len(self.struct_nodes[:, 0])

    def load_modes(self):
        self.Kmodal_inv = np.loadtxt('Kmodal_inv.dat')
        self.Eigvec = np.loadtxt('Eigvec.dat')
        self.nEigval = self.Eigvec.shape[1]

    def process_face(self):
        # Process face information into more easily distinguishable dictionary format
        self.face = {}
        self.aero_centres = np.zeros((self.n_f,3))
        for f in range(self.n_f):
            self.face[f] = {}
            for i in range(4):
                index = f * 4 + i
                self.face[f][i] = {}
                self.face[f][i]['id'] = self.face_nodes[index]
                self.face[f][i]['index_id'] = self.node_labels.index(self.face_nodes[index]) + 1
                self.face[f][i]['coord'] = self.aero_nodes[self.node_labels.index(self.face_nodes[index])]
            self.face[f]['norm'] = -np.cross([np.array(self.face[f][0]['coord']) - np.array(self.face[f][1]['coord'])], [np.array(self.face[f][0]['coord']) - np.array(self.face[f][3]['coord'])])[0]
            self.face[f]['unit_norm'] = self.face[f]['norm'] / np.linalg.norm(self.face[f]['norm'])
            self.face[f]['centre'] = (self.face[f][0]['coord'] + self.face[f][1]['coord'] + self.face[f][2]['coord'] + self.face[f][3]['coord']) / 4
            self.aero_centres[f] = self.face[f]['centre']

    def generate_displacement_transfer_matrix(self, r0, polynomial):
        self.H_u = generate_transfer_matrix(self.aero_nodes, self.struct_nodes, r0, polynomial)

    def generate_pressure_transfer_matrix(self, r0, polynomial):
        self.H_p = generate_transfer_matrix(self.aero_centres, self.struct_nodes, r0, polynomial)

    def calculate_pressure_force(self,p):
        pressure_force = np.zeros((self.n_f,3))
        for f in range(self.n_f):
            pressure_force[f, :] = p[f] * self.face[f]['norm']
        # print('Max aero force = {}'.format(np.max(pressure_force)))
        return pressure_force

    def interp_forces(self,F_a):
        F_s = interp_forces(F_a, self.H_p)
        # print('Max structural force = {}'.format(np.max(F_s)))
        return F_s
    
    def interp_displacements(self,U_s):
        U_a = interp_displacements(U_s,self.H_u)
        for i in range(3):
            self.aero_nodes[:,i] = self.aero_nodes[:,i] + U_a[:,i]

        np.savetxt('deformed.csv', self.aero_nodes, delimiter=",")
        return U_a

    def deform_struct(self,F_s):
        # Rearrange F_s to match ATOM coordinate system
        F_s_atom = np.zeros_like(F_s)
        F_s_atom[:,0] = F_s[:,1]
        F_s_atom[:,1] = -F_s[:,2]
        F_s_atom[:,2] = -F_s[:,0]
        # Reshape F_s
        F_s_atom = np.concatenate((F_s_atom, np.zeros_like(F_s_atom)), axis=1)
        F_s_atom = F_s_atom.flatten(order='F')

        # Convert Nodal forces to modal forces
        FModal = np.matmul(np.transpose(self.Eigvec[:, 0:self.nEigval]), F_s_atom[6:])

        QDisp = np.matmul(self.Kmodal_inv, FModal)

        Disp = np.zeros(self.Eigvec.shape[0])

        # Solve modal model
        for i in range(self.nEigval):
            Disp = np.add(Disp, (QDisp[i] * self.Eigvec[:, i]))

        Disp = np.concatenate((np.zeros(6), Disp))

        Disp = np.reshape(Disp, (self.n_s, 6), order='C')

        # Convert the coordinate systems back

        Disp_zcfd = np.zeros_like(Disp)

        Disp_zcfd[:,0] = -Disp[:,2]
        Disp_zcfd[:,1] = Disp[:,0]
        Disp_zcfd[:,2] = -Disp[:,1]

        # Adjust scale factor here
        return(Disp_zcfd * 1.0e-0)


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
    print(H.shape)
    print(F_a.shape)
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
