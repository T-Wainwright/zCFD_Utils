"""
zCFD Wrapper to combine zCFD functionality with generic RBF coupling 

zCFD is a product of Zenotech ltd.

Tom Wainwright 2021

University of Bristol

tom.wainwright@bristol.ac.uk
"""


class zCFD_mesh():
    def __init__(self):
        # Initialise key variables mesh must have
        self.nodes = []         # Aerodynamic nodes (corners of faces)
        self.centres = []       # Aerodynamic face centres
        self.normals = []       # Normals @ face centres (non unit therefore area)
        self.pressure = []

        self.n_nodes = int

    def load_aeromesh(self, fname, fsi_zone):
        # force load aerodata from mesh- useful if testing not in solver
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

    def calculate_pressure_force(self, p):
        pressure_force = np.zeros((self.n_f, 3))
        for f in range(self.n_f):
            pressure_force[f, :] = p[f] * self.face[f]['unit_norm'] * self.face[f]['area']
        # print('Max aero force = {}'.format(np.max(pressure_force)))
        return pressure_force

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