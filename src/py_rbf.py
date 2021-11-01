import numpy as np
import scipy.io
import h5py
import matplotlib.pyplot as plt


# Functions

class UoB_coupling():

    def __init__(self):
        pass

    def load_struct(self, fname):
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
            faceIndex[i + 1] = faceType[i] + faceIndex[i]

        # find faces with tag fsi_zone
        fsi_faces = np.where(faceInfo[:, 0] == fsi_zone)[0]

        self.face = {}
        self.n_f = len(fsi_faces)
        self.aero_centres = np.zeros((self.n_f, 3))

        # process face dictionary
        for f in range(self.n_f):
            self.face[f] = {}
            faceID = fsi_faces[f]
            n_faceNodes = faceType[fsi_faces[f], 0]
            self.face[f]['n_faceNodes'] = n_faceNodes
            for i in range(n_faceNodes):
                self.face[f][i] = {}
                offset = faceIndex[faceID]
                nodeID = faceNodes[offset + i][0]
                self.face[f][i]['nodeID'] = nodeID
                self.face[f][i]['coord'] = nodeVertex[nodeID]
            self.face[f]['norm'] = -np.cross([np.array(self.face[f][0]['coord']) - np.array(self.face[f][1]['coord'])], [np.array(self.face[f][0]['coord']) - np.array(self.face[f][2]['coord'])])[0]
            self.face[f]['unit_norm'] = self.face[f]['norm'] / np.linalg.norm(self.face[f]['norm'])
            self.face[f]['centre'] = np.mean(np.array([self.face[f][i]['coord'] for i in range(self.face[f]['n_faceNodes'])]), axis=0)

            self.aero_centres[f] = self.face[f]['centre']

            if self.face[f]['n_faceNodes'] == 3:
                self.face[f]['area'] = 0.5 * np.linalg.norm(self.face[f]['norm'])
            elif self.face[f]['n_faceNodes'] == 4:
                self.face[f]['area'] = np.linalg.norm(self.face[f]['norm'])

            else:
                print('ERROR! n_faceNodes = {}'.print(self.face[f]['n_faceNodes']))

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

    def generate_displacement_transfer_matrix(self, r0, rbf, polynomial):
        print('Generating displacement fsi transfer matrix')
        self.H_u = generate_transfer_matrix(self.aero_nodes, self.struct_nodes, r0, rbf, polynomial)

    def generate_pressure_transfer_matrix(self, r0, rbf, polynomial):
        print('Generating pressure fsi transfer matrix')
        self.H_p = generate_transfer_matrix(self.aero_centres, self.struct_nodes, r0, rbf, polynomial)

    def calculate_pressure_force(self, p):
        pressure_force = np.zeros((self.n_f, 3))
        for f in range(self.n_f):
            pressure_force[f, :] = p[f] * self.face[f]['unit_norm'] * self.face[f]['area']
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
        Disp = np.reshape(Disp, (self.n_s, 6))

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

            rib_disps[i * 4 + 0, :] = (np.matmul(np.array([1, 0, 0]), R_z(np.pi/2))  + self.struct_nodes[i, :])
            rib_disps[i * 4 + 1, :] = (np.matmul(np.array([-1, 0, 0]), R_z(np.pi/2)) + self.struct_nodes[i, :])
            rib_disps[i * 4 + 2, :] = (np.matmul(np.array([0, 1, 0]), R_z(np.pi/2))  + self.struct_nodes[i, :])
            rib_disps[i * 4 + 3, :] = (np.matmul(np.array([0, -1, 0]), R_z(np.pi/2)) + self.struct_nodes[i, :])

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

    def update_surface(self, U_a):
        # Update surface normals after a deformation
        for i in range(3):
            self.aero_nodes[:, i] = self.aero_nodes[:, i] + U_a[:, i]

        for f in range(self.n_f):
            for i in range(4):
                index = f * 4 + i
                self.face[f][i]['coord'] = self.aero_nodes[self.node_labels.index(self.face_nodes[index])]
            self.face[f]['norm'] = np.cross([np.array(self.face[f][0]['coord']) - np.array(self.face[f][1]['coord'])], [np.array(self.face[f][0]['coord']) - np.array(self.face[f][3]['coord'])])[0]

    def write_face_normals(self, fname):
        f = open(fname, 'w')
        for face in self.face:
            f.write('{} \t {} \t {} \t {} \t {} \t {} \n'.format(self.face[face]['centre'][0], self.face[face]['centre'][1], self.face[face]['centre'][2], self.face[face]['unit_norm'][0], self.face[face]['unit_norm'][1], self.face[face]['unit_norm'][2]))
            # f.write('{} \t {} \t {} \t {} \t {} \t {} \n'.format(self.face[face][1]['coord'][0], self.face[face][1]['coord'][1], self.face[face][1]['coord'][2], self.face[face]['unit_norm'][0], self.face[face]['unit_norm'][1], self.face[face]['unit_norm'][2]))

        f.close()


def generate_transfer_matrix(aero_nodes, struct_nodes, r0, rbf='c2', polynomial=True):
    # returns- H: (n_a,n_s) full transfer matrix between STRUCTURAL NODES and AERODYNAMIC NODES
    n_a = len(aero_nodes[:, 0])
    # n_a = aero_nodes[:,0].size
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

    print('generate block M')

    # Generate block matrix M (and P if polynomial) equations 11 and 12
    for i in range(n_s):
        for j in range(n_s):
            rad = (np.linalg.norm((struct_nodes[i] - struct_nodes[j]))) / r0
            if rad <= 1.0:
                M_ss[i][j] = rbf(rad)
        if polynomial:
            P_s[1:, i] = struct_nodes[i]
        if i % 1000 == 0:
            print(i)

    print('generate block A_as')
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
    plot_radius = 0.5*max([x_range, y_range, z_range])

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
