import numpy as np
import scipy.io
import h5py


# Functions

class UoB_coupling():

    def __init__(self):
        pass

    def load_struct(self, fname):
        print('Loading FE structural data from {}'.format(fname))
        self.BladeFE = scipy.io.loadmat(FE_fname)

        self.struct_nodes = self.BladeFE['BeamAxis_123'][0][0]
        self.n_s = self.struct_nodes.shape[0]

    def load_modes(self, fname):
        print('Loading Modal data from {}'.format(fname))
        self.ModalStruct = scipy.io.loadmat(fname)['ModalStruct']

        self.nEigval = self.ModalStruct['nEigval'][0][0][0][0]
        self.Eigvec = self.ModalStruct['Eigvec'][0][0]
        self.KmodalInv = self.ModalStruct['KmodalInv'][0][0]

    def load_aeromesh(self, fname, fsi_zone):
        print('obtaining face information from h5 mesh file')
        h5mesh = h5py.File(fname + '.h5')

        # extract required data from h5 file
        numFaces = int(h5mesh['mesh'].attrs.get('numFaces'))

        faceInfo = np.array(h5mesh['mesh']['faceInfo'])
        faceType = np.array(h5mesh['mesh']['faceType'])
        faceNodes = np.array(h5mesh['mesh']['faceNodes'])
        nodeVertex = np.array(h5mesh['mesh']['nodeVertex'])

        faceIndex = np.zeros_like(faceInfo[:, 0])

        for i in range(numFaces - 1):
            faceIndex[i + 1] = faceType[i + 1] + faceIndex[i]

        # find faces with tag fsi_zone
        fsi_faces = np.where(faceInfo[:, 0] == fsi_zone)[0]

        self.face = {}
        self.n_f = len(fsi_faces)

        # process face dictionary
        for f in range(self.n_f):
            self.face[f] = {}
            faceID = fsi_faces[f]
            n_faceNodes = faceInfo[fsi_faces[f], 0]
            self.face[f]['n_faceNodes'] = n_faceNodes
            for i in range(n_faceNodes):
                self.face[f][i] = {}
                nodeID = faceNodes[faceIndex[faceID] + i][0]
                self.face[f][i]['nodeID'] = nodeID
                self.face[f][i]['coord'] = nodeVertex[nodeID]
            self.face[f]['norm'] = -np.cross([np.array(self.face[f][0]['coord']) - np.array(self.face[f][1]['coord'])], [np.array(self.face[f][0]['coord']) - np.array(self.face[f][3]['coord'])])[0]
            self.face[f]['norm'] = -np.cross([np.array(self.face[f][0]['coord']) - np.array(self.face[f][1]['coord'])], [np.array(self.face[f][0]['coord']) - np.array(self.face[f][3]['coord'])])[0]
            self.face[f]['unit_norm'] = self.face[f]['norm'] / np.linalg.norm(self.face[f]['norm'])
            self.face[f]['centre'] = np.mean(np.array([self.face[f][i]['coord'] for i in range(self.face[f]['n_faceNodes'])]), axis=0)

        print(self.n_f)

    def process_face(self):
        # Process face information into more easily distinguishable dictionary format
        print('obtaining face information from solver methods')
        self.face = {}
        self.aero_centres = np.zeros((self.n_f, 3))
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

    def calculate_pressure_force(self, p):
        pressure_force = np.zeros((self.n_f, 3))
        for f in range(self.n_f):
            pressure_force[f, :] = p[f] * self.face[f]['norm']
        # print('Max aero force = {}'.format(np.max(pressure_force)))
        return pressure_force

    def interp_forces(self, F_a):
        F_s = interp_forces(F_a, self.H_p)
        # print('Max structural force = {}'.format(np.max(F_s)))
        return F_s

    def interp_displacements(self, U_s):
        U_a = interp_displacements(U_s, self.H_u)
        for i in range(3):
            self.aero_nodes[:, i] = self.aero_nodes[:, i] + U_a[:, i]

        np.savetxt('deformed.csv', self.aero_nodes, delimiter=",")
        return U_a

    def deform_struct(self, F_s):
        # Rearrange F_s to match ATOM coordinate system
        F_s_atom = np.zeros_like(F_s)
        F_s_atom[:, 0] = F_s[:, 1]
        F_s_atom[:, 1] = -F_s[:, 2]
        F_s_atom[:, 2] = -F_s[:, 0]
        # Reshape F_s
        F_s_atom = np.concatenate((F_s_atom, np.zeros_like(F_s_atom)), axis=1)
        F_s_atom = F_s_atom.flatten(order='F')

        # Convert Nodal forces to modal forces
        FModal = np.matmul(np.transpose(Eigvec[:, 0:nEigval]), Fnodal[6:])

        QDisp = np.matmul(self.KmodalInv, FModal)

        Disp = np.zeros(self.Eigvec.shape[0])

        # Solve modal model
        for i in range(self.nEigval):
            Disp = np.add(Disp, (QDisp[i] * self.Eigvec[:, i]))

        Disp = np.concatenate((np.zeros(6), Disp))

        Disp = np.reshape(Disp, (self.n_s, 6), order='C')

        # Convert the coordinate systems back - discontinued since meshes now have correct coordinate system

        # Disp_zcfd = np.zeros_like(Disp)

        # Disp_zcfd[:, 0] = -Disp[:, 2]
        # Disp_zcfd[:, 1] = Disp[:, 0]
        # Disp_zcfd[:, 2] = -Disp[:, 1]

        # Adjust scale factor here
        return(Disp[:, :2] * 1.0e-0)

    def write_deformed_struct(self, U_s):
        # Dump out record of deformed structure
        f = open("deformed_struct.dat", 'w')
        for i in range(self.n_s):
            f.write('{} \t {} \t {}\n'.format(U_s[i, 0], U_s[i, 1], U_s[i, 2]))
        f.close()

    def update_surface(self, U_a):
        # Update surface normals after a deformation
        for i in range(3):
            self.aero_nodes[:, i] = self.aero_nodes[:, i] + U_a[:, i]

        for f in range(self.n_f):
            for i in range(4):
                index = f * 4 + i
                self.face[f][i]['coord'] = self.aero_nodes[self.node_labels.index(self.face_nodes[index])]
            self.face[f]['norm'] = np.cross([np.array(self.face[f][0]['coord']) - np.array(self.face[f][1]['coord'])], [np.array(self.face[f][0]['coord']) - np.array(self.face[f][3]['coord'])])[0]


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
