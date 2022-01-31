import numpy as np
import h5py

""" 
Converter for CBA mesh to zCFD h5 format

Tom Wainwright

University of Bristol 2020

tom.wainwright@bristol.ac.uk

To convert CBA mesh:

mesh = CBA_mesh('path-to-blk-file')
mesh.convert_h5_data()
mesh.write_h5('path-to-h5-file')

Classes:
- CBA_mesh
    -load_cba(fname,V)
    -structure_data()
    -get_common_faces()
    -solve_faces()
    -convert_h5_data(V,Checks,sort_nodes)
    -get_face_allignment(V)
    -check_unassigned_faces(V)
    -check_unassigned_faceNodes(V)
    -check_unassigned_faceBC(V)
    -check_unassigned_cellFace(V)
    -check_cell_references(V)
    -remove_common_nodes(V)
    -check_tag_assignments()
    -writetec(fname,V)
    -write_ht(fname)

- CBA_block
    -order_points()
    -get_corners()
    -get_face_corners(face)
    -get_n_faces()
    -assign_primary_faces(cell_offset,face_offset)
    -get_boundface_ID(a,b,f)
    -assign_boundface(a,b,f,face_ID)
    -translate_BC()
    -get_faceNodes(i,j,k,p)

- zCFD_mesh
    -load_zcfd()
    -writetec()
    -write_results_tec()
    -writeh5()
    -write_boundary_tec()
    -extractHaloFaces()
    -writeLKEconf()
    -rotate_surface()
    -translate_surface()
    -generate_deformation_file()
    -rbf_rotate()
    """


class CBA_mesh():
    def __init__(self, fname='NONE', V=False):
        # Define properties of CBA mesh

        # Integers
        self.n_cell = 0
        self.npts = 0
        self.n_blocks = 0
        self.n_face = 0

        # dictionaries
        self.block = {}

        # logicals
        self.V = V          # Verbosity flag- turn on for debugging purposed

        # If mesh handle is provided, load mesh
        if fname != 'NONE':
            self.load_cba(fname)

    def load_cba(self, fname):
        # Process data path and fname
        self.fname = fname

        # Load CBA mesh

        data = np.loadtxt(fname)

        # Process CBA data into useful structure

        self.n_blocks = int(data[0, 0])

        line_index = 1

        for b in range(self.n_blocks):
            # Create CBA_block class for each block
            self.block[b] = CBA_block()
            self.block[b].blockID = b

            # Load header information
            self.block[b].npts_i = int(data[line_index, 0])
            self.block[b].npts_j = int(data[line_index, 1])
            self.block[b].npts_k = int(data[line_index, 2])

            self.block[b].npts = self.block[b].npts_i * \
                self.block[b].npts_j * self.block[b].npts_k

            line_index = line_index + 1

            # Load body information
            self.block[b].X = data[line_index:line_index +
                                   self.block[b].npts, :]

            line_index = line_index + self.block[b].npts

            # Load footer information
            for i in range(6):
                self.block[b].connectivity[i] = {}
                self.block[b].connectivity[i]['type'] = int(
                    data[line_index, 0])
                self.block[b].connectivity[i]['neighbour'] = int(
                    data[line_index, 1])
                self.block[b].connectivity[i]['orientation'] = int(
                    data[line_index, 2])
                line_index = line_index + 1

            if self.block[b].npts_k == 1:
                # 2D mesh
                self.block[b].npts_k = 2
                self.block[b].X = np.concatenate(
                    (self.block[b].X, self.block[b].X))
                self.block[b].X[self.block[b].npts:, 1] = 1
                self.block[b].npts = self.block[b].npts_i * \
                    self.block[b].npts_j * self.block[b].npts_k

            self.block[b].n_cell = (self.block[b].npts_i - 1) * \
                (self.block[b].npts_j - 1) * (self.block[b].npts_k - 1)
            self.n_cell = self.n_cell + self.block[b].n_cell
            self.npts = self.npts + self.block[b].npts

        if self.V:
            print(self.n_cell)

        # Perform secondary data manupulations
        self.structure_data()
        self.get_common_faces()
        self.solve_faces()

    def structure_data(self):
        # Process data into fully structured [i][j][k] indexing
        for b in range(self.n_blocks):
            self.block[b].order_points()
            self.block[b].translate_BC()

    def get_common_faces(self):
        # Extract dictionary of common faces within multiblock mesh

        self.common_faces = {}

        n_driving = 0
        n_driven = 0

        # A boundary face will be defined twice.
        # The first instance it is defined will be the "Driving" case
        # The second instance it is defined as the "Driven" case

        # Priority is given by block number, then by face number

        for b in range(self.n_blocks):               # Cycle through blocks
            # Cycle through face boundaries
            for i in self.block[b].connectivity:
                if self.block[b].connectivity[i]['type'] == 2:
                    if self.block[b].connectivity[i]['neighbour'] > b + 1:
                        block2 = self.block[b].connectivity[i]['neighbour'] - 1
                        face2 = self.block[b].connectivity[i]['orientation'] - 1

                        self.common_faces[n_driving] = {}
                        self.common_faces[n_driving]['block1'] = b
                        self.common_faces[n_driving]['face1'] = i
                        self.common_faces[n_driving]['block2'] = block2
                        self.common_faces[n_driving]['face2'] = face2

                        n_driving = n_driving + 1

                        self.block[b].connectivity[i]['driving'] = True
                        self.block[block2].connectivity[face2]['driving'] = False

                    if self.block[b].connectivity[i]['neighbour'] < b + 1:
                        n_driven = n_driven + 1

                    if self.block[b].connectivity[i]['neighbour'] == b + 1:
                        if self.block[b].connectivity[i]['orientation'] > i + 1:
                            block2 = self.block[b].connectivity[i]['neighbour'] - 1
                            face2 = self.block[b].connectivity[i]['orientation'] - 1
                            self.common_faces[n_driving] = {}
                            self.common_faces[n_driving]['block1'] = b
                            self.common_faces[n_driving]['face1'] = i
                            self.common_faces[n_driving]['block2'] = block2
                            self.common_faces[n_driving]['face2'] = face2
                            n_driving = n_driving + 1

                            self.block[b].connectivity[i]['driving'] = True
                            self.block[block2].connectivity[face2]['driving'] = False

                        if self.block[b].connectivity[i]['orientation'] < i + 1:
                            n_driven = n_driven + 1

        if n_driving != n_driven:
            print('ERROR- mismatch in numbers of neighbouring faces')

        self.n_commonface = n_driving

    def solve_faces(self):
        # Find how many faces are in each block, then in the full mesh
        for b in range(self.n_blocks):
            self.block[b].get_n_faces()
            self.n_face = self.n_face + self.block[b].n_face

    def sortnodes(self, find, replace):
        replaceNodes = np.where(self.faceNodes == find)
        self.faceNodes[replaceNodes] = replace

    def convert_h5_data(self, Checks=False, sort_nodes=False):
        # Function to actually run the conversion of meshes
        cell_ID = 0
        face_ID = 0

        self.get_face_allignment()

        # Assign primary faces within blocks
        for b in range(self.n_blocks):
            (cell_ID, face_ID) = self.block[b].assign_primary_faces(
                cell_ID, face_ID)
        if self.V:
            print('Cells assigned: {} \t Faces assigned: {}'.format(cell_ID, face_ID))

        # Solve block boundaries
        for f in self.common_faces:
            # Get number of points to index through
            block1 = self.common_faces[f]['block1']
            block2 = self.common_faces[f]['block2']
            face1 = self.common_faces[f]['face1']
            face2 = self.common_faces[f]['face2']

            if face2 == 0 or face2 == 1:
                npts_a = self.block[block2].npts_j
                npts_b = self.block[block2].npts_k
            elif face2 == 2 or face2 == 3:
                npts_a = self.block[block2].npts_i
                npts_b = self.block[block2].npts_k
            elif face2 == 4 or face2 == 5:
                npts_a = self.block[block2].npts_i
                npts_b = self.block[block2].npts_j

            if self.V:
                print('MB_faceID: {} block1: {} block2: {} face1: {} face2: {}'.format(
                    f, block1, block2, face1, face2))
                print('npts_a: {} npts_b: {}'.format(npts_a, npts_b))

            # Get indexes for driven faces
            for a in range(npts_a - 1):
                for b in range(npts_b - 1):
                    face_ID = self.block[block1].get_boundface_ID(a, b, face1)
                    if self.common_faces[f]['ax1']['reversed']:
                        a2 = npts_a - 2 - a
                    else:
                        a2 = a
                    if self.common_faces[f]['bx1']['reversed']:
                        b2 = npts_b - 2 - b
                    else:
                        b2 = b

                    self.block[block2].assign_boundface(a2, b2, face2, face_ID)

        # Create h5 dataset arrays

        self.faceCell = np.ones([self.n_face, 2], dtype=int) * -100
        self.cellFace = np.ones([self.n_cell * 6], dtype=int) * -100
        self.faceInfo = np.zeros([self.n_face, 2], dtype=int)
        self.faceBC = np.ones([self.n_face], dtype=int) * -100
        self.faceNodes = np.ones([(self.n_face) * 4], dtype=int) * -100
        self.faceType = np.ones([self.n_face], dtype=int) * 4

        # assign faceCell dataset - Full cell index
        for b in range(self.n_blocks):
            for i in range(self.block[b].npts_i - 1):
                for j in range(self.block[b].npts_j - 1):
                    for k in range(self.block[b].npts_k - 1):
                        for f in range(6):
                            face_ID = self.block[b].cell_face[i][j][k][f]
                            cell_ID = self.block[b].cell_face[i][j][k]['cell_ID']

                            if self.faceCell[face_ID, 0] < 0:
                                self.faceCell[face_ID, 0] = cell_ID
                            else:
                                self.faceCell[face_ID, 1] = cell_ID

                            self.cellFace[(cell_ID * 6) + f] = face_ID

        # assign halo cells - full face index

        nhalo = 0

        for f in range(self.n_face):
            if self.faceCell[f, 1] < 0:
                self.faceCell[f, 1] = self.n_cell + nhalo
                nhalo = nhalo + 1
        if self.V:
            print('Halo cells assigned: {}'.format(nhalo))

        # assign boundary conditions - block-face index

        nhalo_face = 0
        point_offset = 0

        for b in range(self.n_blocks):
            for f in self.block[b].face_BC:
                self.faceBC[f] = self.block[b].face_BC[f]
                self.faceInfo[f, 0] = self.block[b].face_info[f]
                if self.block[b].face_BC[f] != 0:
                    nhalo_face = nhalo_face + 1
                for p in range(4):
                    self.faceNodes[f * 4 +
                                   p] = int(self.block[b].face_nodes[f][p] + point_offset)
            point_offset = point_offset + self.block[b].npts
        if self.V:
            print('Halo faces expected: {}'.format(nhalo_face))

        # create nodeVertex dataset - block index
        for b in range(self.n_blocks):
            if b == 0:
                self.nodeVertex = self.block[b].X
            else:
                self.nodeVertex = np.concatenate(
                    (self.nodeVertex, self.block[b].X))

        if sort_nodes:
            self.remove_common_nodes(V)

        # Run Checks
        if Checks:
            self.check_unassigned_cellFace(V)
            self.check_unassigned_faces(V)
            self.check_unassigned_faceBC(V)
            self.check_unassigned_faceNodes(V)

    def get_face_allignment(self):
        # Check the allignement of primary and secondary indexing axis for common faces
        problem_axis = 0
        for f in range(self.n_commonface):
            if self.V:
                print('MB face number: {}'.format(f))
            block1 = self.common_faces[f]['block1']
            face1 = self.common_faces[f]['face1']
            block2 = self.common_faces[f]['block2']
            face2 = self.common_faces[f]['face2']

            if self.V:
                print('block1: {} \tface1: {} \tblock2: {} \tface2: {}'.format(
                    block1, face1, block2, face2))

            face1_corners = self.block[block1].get_face_corners(face1)
            face2_corners = self.block[block2].get_face_corners(face2)

            # Axis for driving faces - A is primary B is secondary

            ax1 = face1_corners[0, :] - face1_corners[1, :]
            bx1 = face1_corners[0, :] - face1_corners[3, :]

            # Axis for driven faces - A is primary B is secondary
            ax2 = face2_corners[0, :] - face2_corners[1, :]
            bx2 = face2_corners[0, :] - face2_corners[3, :]
            cx2 = face2_corners[3, :] - face2_corners[2, :]
            dx2 = face2_corners[1, :] - face2_corners[2, :]

            face1_axes = {'ax1': ax1, 'bx1': bx1}
            face2_axes = {'ax2': ax2, 'bx2': bx2, 'cx2': cx2, 'dx2': dx2}

            if self.V:
                print(face1_axes)

            axis_colinear = 0
            axis_reversed = 0

            for i in face1_axes:
                for j in face2_axes:
                    cross = np.cross(face1_axes[i], face2_axes[j])
                    dot = np.dot(face1_axes[i], face2_axes[j])

                    if np.linalg.norm(cross) < 0.00001:
                        axis_colinear = axis_colinear + 1
                        self.common_faces[f][i] = {}
                        self.common_faces[f][i]['aligned'] = j
                        if dot > 0:
                            msg = 'alligned'
                            self.common_faces[f][i]['reversed'] = False
                        elif dot < 0:
                            msg = 'reversed'
                            self.common_faces[f][i]['reversed'] = True
                        if self.V:
                            print(
                                'Colinear: {}, {} \t direction: {}'.format(i, j, msg))

            if axis_colinear != 2:
                problem_axis = problem_axis + 1

        if problem_axis != 0:
            if self.V:
                print('Check axis orientations')
                print(problem_axis)

    def check_unassigned_faces(self):
        # Check if any faces assigned for blocks have not been assigned
        if self.V:
            print('Checking for unassigned faces...')
        unassigned_faces = 0
        for b in range(self.n_blocks):
            for i in range(self.block[b].npts_i - 1):
                for j in range(self.block[b].npts_j - 1):
                    for k in range(self.block[b].npts_k - 1):
                        for f in range(6):
                            if self.block[b].cell_face[i][j][k][f] == 'hold':
                                unassigned_faces = unassigned_faces + 1
        if self.V:
            print('{} faces unassigned'.format(unassigned_faces))

    def check_unassigned_faceNodes(self):
        # Check if any faceNodes have not been assigned
        if self.V:
            print('Checking for unassigned faceNodes...')
        unassigned_faceNodes = 0
        for f in self.faceNodes:
            if f < 0:
                unassigned_faceNodes = unassigned_faceNodes + 1
        if self.V:
            print('{} faceNodes unassigned'.format(unassigned_faceNodes))

    def check_unassigned_faceBC(self):
        # Check if any faceBC entries are unassigned
        if self.V:
            print('Checking for unassigned faceBC...')
        unassigned_faceBC = 0
        for f in self.faceBC:
            if f < 0:
                unassigned_faceBC = unassigned_faceBC + 1
        if self.V:
            print('{} faceBC unassigned'.format(unassigned_faceBC))

    def check_unassigned_cellFace(self):
        # Check if any cellFace entries are unassigned
        if self.V:
            print('Checking for unassigned cellFaces')
        unassigned_cellFace = 0
        for f in self.cellFace:
            if f < 0:
                unassigned_cellFace = unassigned_cellFace + 1
        if self.V:
            print('{} cellFaces unassigned'.format(unassigned_cellFace))

    def check_cell_references(self):
        # Check how many times each cell is referenced in faces- should be 6 for each cell
        if self.V:
            print('Checking cell references')
        cell_references = np.zeros(self.n_cell, dtype=int)
        for f in range(self.n_face):
            cell_references[self.faceCell[f, 0]
                            ] = cell_references[self.faceCell[f, 0]] + 1
            if self.faceCell[f, 1] < self.n_cell:
                cell_references[self.faceCell[f, 1]
                                ] = cell_references[self.faceCell[f, 1]] + 1

    def remove_common_nodes(self):
        # Nodes on block boundaries will be defined twice, remove the second existence of them, and change indexing
        if self.V:
            print('Removing common nodes')

        node_references = np.zeros_like(self.nodeVertex[:, 0])
        presort_nodes = len(self.nodeVertex[:, 0])

        for fn in self.faceNodes:
            node_references[fn] = node_references[fn] + 1

        unique, counts = np.unique(node_references, return_counts=True)

        if self.V:
            print('Number of node references (pre-sort)')
            print(dict(zip(unique, counts)))

        unique, indices = np.unique(
            self.nodeVertex, axis=0, return_inverse=True)

        faceNodes_sorted = np.zeros_like(self.faceNodes)

        for fn in range(len(self.faceNodes)):
            faceNodes_sorted[fn] = indices[self.faceNodes[fn]]

        self.faceNodes = faceNodes_sorted
        self.nodeVertex = unique
        postsort_nodes = len(self.nodeVertex[:, 0])

        node_references = np.zeros_like(self.nodeVertex[:, 0])

        for fn in self.faceNodes:
            node_references[fn] = node_references[fn] + 1

        unique, counts = np.unique(node_references, return_counts=True)
        if self.V:
            print('Number of node references (post-sort)')
            print(dict(zip(unique, counts)))

            print('{} Common nodes removed'.format(
                presort_nodes - postsort_nodes))

    def check_tag_assignements(self):
        # Check how many face tags are assigned for each relevent datasets
        print('n_faces: {}'.format(self.n_face))
        print('faceBC tags:')
        faceBC_tags, faceBC_count = np.unique(self.faceBC, return_counts=True)
        print(dict(zip(faceBC_tags, faceBC_count)))

        print('faceInfo tags:')
        faceInfo_tags, faceInfo_count = np.unique(
            self.faceInfo[:, 0], return_counts=True)
        print(dict(zip(faceInfo_tags, faceInfo_count)))

        print('faceType tags:')
        faceType_tags, faceType_count = np.unique(
            self.faceType, return_counts=True)
        print(dict(zip(faceType_tags, faceType_count)))

    def write_zCFD_tec(self, fname='NONE'):
        # Process fname
        if fname == 'NONE':
            fname = self.fname + '.h5.plt'
        # Write ZCFD mesh to tecplot FEPOLYHEDRON FORMAT
        if self.V:
            print('Writing tecplot mesh file: {}'.format(fname))
        n_v = np.size(self.nodeVertex[:, 0])
        n_c = self.n_cell
        n_f = self.n_face
        n_fnodes = np.size(self.faceNodes)

        fout = open(fname, "w")
        if self.V:
            print('Writing Header Information')
        fout.write("VARIABLES= \"X\" \"Y\" \"Z\"\n")
        fout.write("ZONE \n")
        # Number of Nodes
        fout.write("NODES = {} \n".format(n_v))
        # Number of faces
        fout.write("FACES = {} \n".format(n_f))
        fout.write("TOTALNUMFACENODES = {} \n".format(
            n_fnodes))    # Number of nodes in faces
        # Number of connected boundary faces (0)
        fout.write("NUMCONNECTEDBOUNDARYFACES = 0 \n")
        # Number of connected zones (0)
        fout.write("TOTALNUMBOUNDARYCONNECTIONS = 0 \n")
        # Number of cells
        fout.write("ELEMENTS = {} \n".format(n_c))
        # Data formatting- must be block for FEPOLYHEDRON
        fout.write("DATAPACKING = BLOCK \n")
        # Mesh type- FE polyhedron for zCFD
        fout.write("ZONETYPE = FEPOLYHEDRON \n")

        if self.V:
            print('Writing Node Vertex Points')
        fout.write('# i Vertex Locations \n')
        for i in range(n_v):
            fout.write("{} \n".format(self.nodeVertex[i, 0]))
        fout.write('# j Vertex Locations \n')
        for i in range(n_v):
            fout.write("{} \n".format(self.nodeVertex[i, 1]))
        fout.write('# k Vertex Locations \n')
        for i in range(n_v):
            fout.write("{} \n".format(self.nodeVertex[i, 2]))

        if self.V:
            print('Writing Face Info')
        fout.write('# Number of points per face \n')
        for i in range(n_f):
            fout.write("{} \n".format(self.faceType[i]))

        if self.V:
            print('Writing Face Nodes')
        fout.write('# Nodes making up each face \n')
        for i in range(n_f):
            n_points = int(self.faceType[i])
            for j in range(n_points):
                index = i * n_points + j
                fout.write("{} ".format(self.faceNodes[index] + 1))
            fout.write("\n")

        if self.V:
            print('Writing Face Cell Interfaces')
        fout.write('# Left Cells \n')
        for i in range(n_f):
            fout.write("{} \n".format(int(self.faceCell[i, 0] + 1)))
        fout.write('# Right Cells \n')
        for i in range(n_f):
            if self.faceCell[i, 1] < n_c:
                fout.write("{} \n".format(int(self.faceCell[i, 1] + 1)))
            elif self.faceCell[i, 1] >= n_c:
                fout.write("0 \n")

        if self.V:
            print('tecplot file written successfully')
        return

    def write_h5(self, fname='NONE'):
        # Process fname
        if fname == 'NONE':
            fname = self.fname + '.h5'

        # Create .h5 file
        f = h5py.File(fname, "w")
        h5mesh = f.create_group("mesh")

        # Assign attributes
        h5mesh.attrs.create("numFaces", self.n_face, shape=(1, 1))
        h5mesh.attrs.create("numCells", self.n_cell, shape=(1, 1,))

        # Assign datasets
        h5mesh.create_dataset("cellFace", data=self.cellFace)
        h5mesh.create_dataset("faceBC", data=self.faceBC,
                              shape=(self.n_face, 1))
        h5mesh.create_dataset("faceCell", data=self.faceCell)
        h5mesh.create_dataset("faceInfo", data=self.faceInfo)
        h5mesh.create_dataset(
            "faceNodes", data=self.faceNodes, shape=(self.n_face * 4, 1))
        h5mesh.create_dataset("faceType", data=self.faceType)
        h5mesh.create_dataset("nodeVertex", data=self.nodeVertex)
        return

    def write_multiblock_tec(self, fname):
        # Convert CBA mesh to tecplot visualisation format
        if self.V:
            print('Generating Structured tecplot file')
        fout = open(fname, "w")
        fout.write('VARIABLES = "X" "Y" "Z" \n')
        for i in self.block:
            fout.write('ZONE I= {} J= {} K= {} F=POINT \n'.format(
                self.block[i].npts_i, self.block[i].npts_j, self.block[i].npts_k))
            for j in range(self.block[i].npts):
                fout.write('{:.15f} \t {:.15f} \t {:.15f} \n'.format(
                    self.block[i].X[j, 0], self.block[i].X[j, 1], self.block[i].X[j, 2]))
        return

    def writeCBA(self, fname):
        # Write saved mesh back to CBA format
        if self.V:
            print('Generating blk mesh file')
        fout = open(fname, "w")
        fout.write('{} \t {} \t {}\n'.format(self.n_blocks, 1, 2.5))
        for i in self.block:
            fout.write('{} \t {} \t {}\n'.format(
                self.block[i].npts_i, self.block[i].npts_j, self.block[i].npts_k))
            for j in range(self.block[i].npts):
                fout.write('{:.15f} \t {:.15f} \t {:.15f} \n'.format(
                    self.block[i].X[j, 0], self.block[i].X[j, 1], self.block[i].X[j, 2]))
            for j in range(6):
                fout.write('{} \t {} \t {} \n'.format(int(self.block[i].connectivity[j]['type']), int(
                    self.block[i].connectivity[j]['neighbour']), int(self.block[i].connectivity[j]['orientation'])))
        return

    def write_p3d(self, fname):
        # Use PLOT3D format to import CBA meshes into pointwise
        f = open(fname, "w")

        f.write('{}\n'.format(self.n_blocks))
        for b in range(self.n_blocks):
            f.write('{} \t {} \t {}\n'.format(
                self.block[b].npts_i, self.block[b].npts_j, self.block[b].npts_k))

        for b in range(self.n_blocks):
            for i in range(self.block[b].npts):
                f.write('{}\n'.format(self.block[b].X[i, 0]))
            for i in range(self.block[b].npts):
                f.write('{}\n'.format(self.block[b].X[i, 1]))
            for i in range(self.block[b].npts):
                f.write('{}\n'.format(self.block[b].X[i, 2]))
        f.close()
        return


class CBA_block():
    def __init__(self):
        # Define properties of CBA block

        # Integers

        self.n_cell = 0
        self.npts = 0
        self.npts_i = 0
        self.npts_j = 0
        self.npts_k = 0
        self.blockID = 0

        # Arrays
        self.X = []

        # Dictionaries
        self.connectivity = {}

    def order_points(self):
        # Re-structure points to 3D array
        self.pts = np.zeros([self.npts_i, self.npts_j, self.npts_k, 3])

        index = 0

        for k in range(self.npts_k):
            for j in range(self.npts_j):
                for i in range(self.npts_i):
                    self.pts[i, j, k, 0] = self.X[index, 0]
                    self.pts[i, j, k, 1] = self.X[index, 1]
                    self.pts[i, j, k, 2] = self.X[index, 2]
                    index = index + 1

    def get_corners(self):
        # Get corner points of block
        corners = np.zeros([8, 3])
        index = 0

        # imin,jmin,kmin
        # imax,jmin,kmin
        # imin,jmax,kmin
        # imax,jmax,kmin
        # imin,jmin,kmax
        # imax,jmin,kmax
        # imin,jmax,kmax
        # imax,jmax,kmax

        for k in range(2):
            for j in range(2):
                for i in range(2):
                    corners[index, 0] = self.pts[i *
                                                 (self.npts_i - 1), j * (self.npts_j - 1), k * (self.npts_k - 1), 0]
                    corners[index, 1] = self.pts[i *
                                                 (self.npts_i - 1), j * (self.npts_j - 1), k * (self.npts_k - 1), 1]
                    corners[index, 2] = self.pts[i *
                                                 (self.npts_i - 1), j * (self.npts_j - 1), k * (self.npts_k - 1), 2]

                    index = index + 1

        return(corners)

    def get_face_corners(self, face):
        # extract corners from a specific face

        corners = self.get_corners()

        face_corners = np.zeros([4, 3])

        if face == 0:       # i min
            f_V = [0, 2, 6, 4]
        elif face == 1:     # i max
            f_V = [1, 3, 7, 5]
        elif face == 2:     # j min
            f_V = [0, 1, 5, 4]
        elif face == 3:     # j max
            f_V = [2, 3, 7, 6]
        elif face == 4:     # k min
            f_V = [0, 1, 3, 2]
        elif face == 5:     # k max
            f_V = [4, 5, 7, 6]

        for i in range(4):
            face_corners[i, :] = corners[f_V[i], :]

        return(face_corners)

    def get_n_faces(self):
        # Get the number of faces in block, taking into accound driven faces

        # Number of faces in plane (jk is i face)
        n_face_jk = (self.npts_j - 1) * (self.npts_k - 1)
        n_face_ik = (self.npts_i - 1) * (self.npts_k - 1)
        n_face_ij = (self.npts_i - 1) * (self.npts_j - 1)

        i_planes = self.npts_i
        j_planes = self.npts_j
        k_planes = self.npts_k

        for f in self.connectivity:
            if self.connectivity[f]['type'] == 2:
                if not self.connectivity[f]['driving']:
                    if f == 0 or f == 1:
                        i_planes = i_planes - 1
                    elif f == 2 or f == 3:
                        j_planes = j_planes - 1
                    elif f == 4 or f == 5:
                        k_planes = k_planes - 1

        self.n_face_i = n_face_jk * i_planes
        self.n_face_j = n_face_ik * j_planes
        self.n_face_k = n_face_ij * k_planes

        self.n_face = self.n_face_i + self.n_face_j + self.n_face_k

        self.nbound_face = (n_face_jk + n_face_ik + n_face_ij) * 2

    def assign_primary_faces(self, cell_offset, face_offset):
        # Assign all non-driven faces to the block
        self.cell_face = {}                       # Faces on each cell
        self.face_BC = {}
        self.face_nodes = {}
        self.face_info = {}

        cell_ID = 0
        face_ID = 0
        boundary_faces = 0
        iface_ID = 0

        # Logical array to dictate boundary state
        boundary_conditions = [0, self.npts_i - 2,
                               0, self.npts_j - 2, 0, self.npts_k - 2]
        # Logical array to dictate which internal faces should be assigned uniquely
        internal_conditions = [False, True, False, True, False, True]

        for i in range(self.npts_i - 1):
            self.cell_face[i] = {}
            for j in range(self.npts_j - 1):
                self.cell_face[i][j] = {}
                for k in range(self.npts_k - 1):
                    # Number cells
                    self.cell_face[i][j][k] = {}
                    self.cell_face[i][j][k]['cell_ID'] = cell_ID + \
                        cell_offset          # Assign next cellID

                    # Number faces

                    # identify if at boundary- then if unique face
                    # Position of the cell within the block
                    position = [i, i, j, j, k, k]
                    # Indexed corners of each cell
                    corners = [i, i + 1, j, j + 1, k, k + 1]

                    for p in range(6):
                        if position[p] == boundary_conditions[p]:
                            # Face p is a boundary face
                            boundary_faces = boundary_faces + 1
                            # Face is internal boundary face
                            if self.connectivity[p]['type'] == 2:
                                # Face is driven internal boundary face
                                if not self.connectivity[p]['driving']:
                                    self.cell_face[i][j][k][p] = 'hold'
                                else:
                                    # Face is driving internal boundary face
                                    self.cell_face[i][j][k][p] = face_ID + \
                                        face_offset
                                    self.face_BC[face_ID + face_offset] = 0
                                    self.face_info[face_ID + face_offset] = 0
                                    self.face_nodes[face_ID +
                                                    face_offset] = self.get_faceNodes(i, j, k, p)

                                    face_ID = face_ID + 1

                            else:
                                # Face is boundary face
                                self.cell_face[i][j][k][p] = face_ID + \
                                    face_offset
                                self.face_BC[face_ID +
                                             face_offset] = self.connectivity[p]['BC_translated']
                                self.face_info[face_ID +
                                               face_offset] = self.connectivity[p]['FI_translated']
                                self.face_nodes[face_ID +
                                                face_offset] = self.get_faceNodes(i, j, k, p)

                                face_ID = face_ID + 1

                        # Face is internal driving (max) face
                        elif internal_conditions[p]:
                            self.cell_face[i][j][k][p] = face_ID + face_offset
                            self.face_BC[face_ID + face_offset] = 0
                            self.face_info[face_ID + face_offset] = 0
                            self.face_nodes[face_ID +
                                            face_offset] = self.get_faceNodes(i, j, k, p)

                            face_ID = face_ID + 1

                        # Face is internal driven (min) face
                        elif p == 0:
                            self.cell_face[i][j][k][0] = self.cell_face[i - 1][j][k][1]
                        elif p == 2:
                            self.cell_face[i][j][k][2] = self.cell_face[i][j - 1][k][3]
                        elif p == 4:
                            self.cell_face[i][j][k][4] = self.cell_face[i][j][k - 1][5]

                    cell_ID = cell_ID + 1

        if face_ID != self.n_face:
            print('Mismatch in face numbers: {} assigned, {} expected'.format(
                face_ID, self.n_face))
            print('Difference of {}'.format(self.n_face - face_ID))
        if cell_ID != self.n_cell:
            print('Mismatch in cell numbers: {} assigned, {} expected'.format(
                cell_ID, self.n_cell))
            print('Difference of {}'.format(self.n_cell - cell_ID))

        return (cell_ID + cell_offset), (face_ID + face_offset)

    def get_boundface_ID(self, a, b, f):
        # Get the ID for a boundary face (f) with primary index a and secondary index b
        face_ID = 0

        if f == 0:
            face_ID = self.cell_face[0][a][b][0]
        elif f == 1:
            face_ID = self.cell_face[self.npts_i - 2][a][b][1]
        elif f == 2:
            face_ID = self.cell_face[a][0][b][2]
        elif f == 3:
            face_ID = self.cell_face[a][self.npts_j - 2][b][3]
        elif f == 4:
            face_ID = self.cell_face[a][b][0][4]
        elif f == 5:
            face_ID = self.cell_face[a][b][self.npts_k - 2][5]

        return face_ID

    def assign_boundface(self, a, b, f, face_ID):
        # Assign face_ID to boundary face f with primary index a and secondary index b

        # Check we're not pushing a face to a non-driven face
        if self.connectivity[f]['driving']:
            print('ERROR- PUSHING TO NON DRIVEN FACE')
            print('Pushing to block: {} face: {}'.format(self.blockID, f))
        if f == 0:
            self.cell_face[0][a][b][0] = face_ID
        elif f == 1:
            self.cell_face[self.npts_i - 2][a][b][1] = face_ID
        elif f == 2:
            self.cell_face[a][0][b][2] = face_ID
        elif f == 3:
            self.cell_face[a][self.npts_j - 2][b][3] = face_ID
        elif f == 4:
            self.cell_face[a][b][0][4] = face_ID
        elif f == 5:
            self.cell_face[a][b][self.npts_k - 2][5] = face_ID

    def resolve_cellNodes(self):
        cell_nodes = {}
        for i in range(self.npts_i - 1):
            for j in range(self.npts_j - 1):
                for k in range(self.npts_k - 1):
                    for p in range(6):
                        cell_nodes[p] = self.cell_face[i][j][k][p]['nodes']
                        # print(cell_nodes)

    def translate_BC(self):
        # Translate boundary conditions

        # CBA format:
        # -2 = 2D wall (symmetry)
        # -1 = Aerodynamic surface
        # 0 = wall
        # 1 = Farfield
        # 2 = Internal face
        # 3 = Periodic Downstream
        # 4 = Periodic Upstream

        # zCFD format:
        # 0 = NONE
        # 2 = Interior
        # 3 = Wall
        # 4 = Inflow
        # 5 = Outflow
        # 7 = Symmetry
        # 9 = Farfield
        # 12 = Periodic
        # 13 = Accoustic wall source

        # Fluid zones
        # 0 = NONE
        # 2 = Farfield
        # 3 = Slip Wall
        # 4 = Aerodynamic surface
        # 5 = Periodic Downstream
        # 6 = Periodic Upstream

        BC_dict = {-2: 7, -1: 3, 0: 3, 1: 9, 2: 0, 3: 12, 4: 12}
        FI_dict = {-2: 7, -1: 4, 0: 3, 1: 2, 2: 0, 3: 5, 4: 6}
        for f in range(6):
            self.connectivity[f]['BC_translated'] = BC_dict[self.connectivity[f]['type']]
            self.connectivity[f]['FI_translated'] = FI_dict[self.connectivity[f]['type']]

    def get_faceNodes(self, i, j, k, p):
        # Get the node index for face p with index i,j,k
        if p == 0:
            nv1 = i + j * self.npts_i + k * self.npts_i * self.npts_j
            nv2 = i + j * self.npts_i + (k + 1) * self.npts_i * self.npts_j
            nv3 = i + (j + 1) * self.npts_i + (k + 1) * \
                self.npts_i * self.npts_j
            nv4 = i + (j + 1) * self.npts_i + k * self.npts_i * self.npts_j
        elif p == 1:
            nv1 = (i + 1) + j * self.npts_i + k * self.npts_i * self.npts_j
            nv2 = (i + 1) + (j + 1) * self.npts_i + \
                k * self.npts_i * self.npts_j
            nv3 = (i + 1) + (j + 1) * self.npts_i + \
                (k + 1) * self.npts_i * self.npts_j
            nv4 = (i + 1) + j * self.npts_i + \
                (k + 1) * self.npts_i * self.npts_j
        elif p == 2:
            nv1 = i + j * self.npts_i + k * self.npts_i * self.npts_j
            nv2 = (i + 1) + j * self.npts_i + k * self.npts_i * self.npts_j
            nv3 = (i + 1) + j * self.npts_i + \
                (k + 1) * self.npts_i * self.npts_j
            nv4 = i + j * self.npts_i + (k + 1) * self.npts_i * self.npts_j
        elif p == 3:
            nv1 = i + (j + 1) * self.npts_i + k * self.npts_i * self.npts_j
            nv2 = i + (j + 1) * self.npts_i + (k + 1) * \
                self.npts_i * self.npts_j
            nv3 = (i + 1) + (j + 1) * self.npts_i + \
                (k + 1) * self.npts_i * self.npts_j
            nv4 = (i + 1) + (j + 1) * self.npts_i + \
                k * self.npts_i * self.npts_j
        elif p == 4:
            nv1 = i + j * self.npts_i + k * self.npts_i * self.npts_j
            nv2 = i + (j + 1) * self.npts_i + k * self.npts_i * self.npts_j
            nv3 = (i + 1) + (j + 1) * self.npts_i + \
                k * self.npts_i * self.npts_j
            nv4 = (i + 1) + j * self.npts_i + k * self.npts_i * self.npts_j
        elif p == 5:
            nv1 = i + j * self.npts_i + (k + 1) * self.npts_i * self.npts_j
            nv2 = (i + 1) + j * self.npts_i + \
                (k + 1) * self.npts_i * self.npts_j
            nv3 = (i + 1) + (j + 1) * self.npts_i + \
                (k + 1) * self.npts_i * self.npts_j
            nv4 = i + (j + 1) * self.npts_i + (k + 1) * \
                self.npts_i * self.npts_j

        # Return order important with negative volumes
        return [nv4, nv3, nv2, nv1]


class zCFD_mesh:
    # Class containing data in zCFD mesh format
    def __init__(self, fname='NONE'):
        # H5 Attributes
        self.numFaces = 0
        self.numCells = 0
        # H5 Datasets
        self.cellFace = np.array([])
        self.faceBC = np.array([])
        self.faceCell = np.array([])
        self.faceInfo = np.array([])
        self.faceNodes = np.array([])
        self.faceType = np.array([])
        self.nodeVertex = np.array([])

        # Logicals
        self.V = False

        if fname != 'NONE':
            self.load_zcfd(fname)

    def load_zcfd(self, fname, V=False):
        self.V = V
        # Load zCFD h5 unstructured mesh
        if self.V:
            print('Loading zCFD mesh: {}'.format(fname))
        f = h5py.File(fname, "r")
        g = f.get('mesh')

        # Get attributes
        self.numFaces = int(g.attrs.get('numFaces'))
        self.numCells = int(g.attrs.get('numCells'))

        # Get data sets
        self.cellZone = np.array(g.get('cellZone'))
        self.cellFace = np.array(g.get('cellFace'))
        self.cellType = np.array(g.get('cellType'))
        self.faceBC = np.array(g.get('faceBC'))
        self.faceCell = np.array(g.get('faceCell'))
        self.faceInfo = np.array(g.get('faceInfo'))
        self.faceNodes = np.array(g.get('faceNodes'))
        self.faceType = np.array(g.get('faceType'))
        self.nodeVertex = np.array(g.get('nodeVertex'))

        # create additional faceIndex dataset:
        self.faceIndex = np.zeros_like(self.faceType)
        for i in range(self.numFaces - 1):
            self.faceIndex[i + 1] = self.faceType[i + 1] + self.faceIndex[i]

        if self.V:
            print('zCFD mesh successfully loaded ')
            print('nCells= {} \t nFaces= {}'.format(
                self.numCells, self.numFaces))

        return

    def writetec(self, fname):
        # Write ZCFD mesh to tecplot FEPOLYHEDRON FORMAT
        if self.V:
            print('Writing tecplot mesh file: {}'.format(fname))

        n_v = np.size(self.nodeVertex[:, 0])
        n_c = self.numCells
        n_f = self.numFaces
        n_fnodes = np.size(self.faceNodes)

        fout = open(fname, "w")

        # File header information
        if self.V:
            print('Writing Header Information')

        fout.write("VARIABLES= \"X\" \"Y\" \"Z\"\n")
        fout.write("ZONE \n")
        # Number of Nodes
        fout.write("NODES = {} \n".format(n_v))
        # Number of faces
        fout.write("FACES = {} \n".format(n_f))
        fout.write("TOTALNUMFACENODES = {} \n".format(
            n_fnodes))    # Number of nodes in faces
        # Number of connected boundary faces (0)
        fout.write("NUMCONNECTEDBOUNDARYFACES = 0 \n")
        # Number of connected zones (0)
        fout.write("TOTALNUMBOUNDARYCONNECTIONS = 0 \n")
        # Number of cells
        fout.write("ELEMENTS = {} \n".format(n_c))
        # Data formatting- must be block for FEPOLYHEDRON
        fout.write("DATAPACKING = BLOCK \n")
        # Mesh type- FE polyhedron for zCFD
        fout.write("ZONETYPE = FEPOLYHEDRON \n")

        # File body information
        if self.V:
            print('Writing Node Vertex Points')
        fout.write('# i Vertex Locations \n')
        for i in range(n_v):
            fout.write("{} \n".format(self.nodeVertex[i, 0]))
        fout.write('# j Vertex Locations \n')
        for i in range(n_v):
            fout.write("{} \n".format(self.nodeVertex[i, 1]))
        fout.write('# k Vertex Locations \n')
        for i in range(n_v):
            fout.write("{} \n".format(self.nodeVertex[i, 2]))

        if self.V:
            print('Writing Face Info')
        fout.write('# Number of points per face \n')
        for i in range(n_f):
            fout.write("{} \n".format(self.faceType[i]))

        if self.V:
            print('Writing Face Nodes')
        fout.write('# Nodes making up each face \n')
        index = 0
        for i in range(n_f):
            n_points = int(self.faceType[i])
            for j in range(n_points):
                index = index + j
                fout.write("{} ".format(self.faceNodes[index, 0] + 1))
            fout.write("\n")

        if self.V:
            print('Writing Face Cell Interfaces')
        fout.write('# Left Cells \n')
        for i in range(n_f):
            fout.write("{} \n".format(self.faceCell[i, 0] + 1))
        fout.write('# Right Cells \n')
        for i in range(n_f):
            if self.faceCell[i, 1] < n_c:
                fout.write("{} \n".format(self.faceCell[i, 1] + 1))
            elif self.faceCell[i, 1] >= n_c:
                fout.write("0 \n")

        if self.V:
            print('tecplot file written successfully')

    def write_results_tec(self, fname, results):
        # Write ZCFD mesh to tecplot FEPOLYHEDRON FORMAT
        if self.V:
            print('Writing tecplot mesh file: {}'.format(fname))
        n_v = np.size(self.nodeVertex[:, 0])
        n_c = self.numCells
        n_f = self.numFaces
        n_fnodes = np.size(self.faceNodes)

        fout = open(fname, "w")

        # File header information
        if self.V:
            print('Writing Header Information')
        fout.write(
            "VARIABLES= \"X\" \"Y\" \"Z\" \"P\" \"Vx\" \"Vy\" \"Vz\" \"Density\"\n")
        fout.write("ZONE \n")
        # Number of Nodes
        fout.write("NODES = {} \n".format(n_v))
        # Number of faces
        fout.write("FACES = {} \n".format(n_f))
        fout.write("TOTALNUMFACENODES = {} \n".format(
            n_fnodes))    # Number of nodes in faces
        # Number of connected boundary faces (0)
        fout.write("NUMCONNECTEDBOUNDARYFACES = 0 \n")
        # Number of connected zones (0)
        fout.write("TOTALNUMBOUNDARYCONNECTIONS = 0 \n")
        # Number of cells
        fout.write("ELEMENTS = {} \n".format(n_c))
        # Data formatting- must be block for FEPOLYHEDRON
        fout.write("DATAPACKING = BLOCK \n")
        # Mesh type- FE polyhedron for zCFD
        fout.write("ZONETYPE = FEPOLYHEDRON \n")
        # Where variables are stored- note current limitation that surface values not preserved
        fout.write("VARLOCATION=([4,5,6,7,8]=CELLCENTERED)\n")

        # File body information
        if self.V:
            print('Writing Node Vertex Points')
        fout.write('# i Vertex Locations \n')
        for i in range(n_v):
            fout.write("{} \n".format(self.nodeVertex[i, 0]))
        fout.write('# j Vertex Locations \n')
        for i in range(n_v):
            fout.write("{} \n".format(self.nodeVertex[i, 1]))
        fout.write('# k Vertex Locations \n')
        for i in range(n_v):
            fout.write("{} \n".format(self.nodeVertex[i, 2]))

        for i in range(n_c):
            fout.write("{} \n".format(
                results.dset['solution'][results.dset['globalToLocalIndex'][i]][0]))
        for i in range(n_c):
            fout.write("{} \n".format(
                results.dset['solution'][results.dset['globalToLocalIndex'][i]][1]))
        for i in range(n_c):
            fout.write("{} \n".format(
                results.dset['solution'][results.dset['globalToLocalIndex'][i]][2]))
        for i in range(n_c):
            fout.write("{} \n".format(
                results.dset['solution'][results.dset['globalToLocalIndex'][i]][3]))
        for i in range(n_c):
            fout.write("{} \n".format(
                results.dset['solution'][results.dset['globalToLocalIndex'][i]][4]))

        if self.V:
            print('Writing Face Info')
        fout.write('# Number of points per face \n')
        for i in range(n_f):
            fout.write("{} \n".format(self.faceType[i][0]))

        if self.V:
            print('Writing Face Nodes')
        fout.write('# Nodes making up each face \n')
        for i in range(n_f):
            n_points = int(self.faceType[i])
            for j in range(n_points):
                index = i * n_points + j
                fout.write("{} ".format(self.faceNodes[index, 0] + 1))
            fout.write("\n")

        if self.V:
            print('Writing Face Cell Interfaces')
        fout.write('# Left Cells \n')
        for i in range(n_f):
            fout.write("{} \n".format(self.faceCell[i, 0] + 1))
        fout.write('# Right Cells \n')
        for i in range(n_f):
            if self.faceCell[i, 1] < n_c:
                fout.write("{} \n".format(self.faceCell[i, 1] + 1))
            elif self.faceCell[i, 1] >= n_c:
                fout.write("0 \n")

        if self.V:
            print('tecplot results file written successfully')

    def writeh5(self, fname):
        # Write unstructured data to h5 file
        if self.V:
            print('Writing h5 mesh file: {}'.format(fname))
        f = h5py.File(fname, "w")
        h5mesh = f.create_group("mesh")

        # Write attributes
        h5mesh.attrs.create("numFaces", self.numFaces, shape=(1, 1))
        h5mesh.attrs.create("numCells", self.numCells, shape=(1, 1))

        # Write datasets
        h5mesh.create_dataset("faceBC", data=self.faceBC,
                              shape=(self.numFaces, 1))
        h5mesh.create_dataset("faceCell", data=self.faceCell)
        h5mesh.create_dataset("faceInfo", data=self.faceInfo)
        h5mesh.create_dataset(
            "faceNodes", data=self.faceNodes, shape=(self.numFaces * 4, 1))
        h5mesh.create_dataset(
            "faceType", data=self.faceType, shape=(self.numFaces, 1))
        h5mesh.create_dataset("nodeVertex", data=self.nodeVertex)

        if self.V:
            print('h5 file written successfully')
        return

    def write_boundary_tec(self, fname):
        # Export boundaries from h5 mesh
        if self.V:
            print('Extracting boundary faces')

        # Get surface face ID's
        surface_faceID = np.array(np.where(self.faceInfo[:, 0] != 0))[
            0, :]     # Find index of faces with non-zero face tag
        # Number of boundary faces
        n_face = len(surface_faceID)
        # array to store the tag of specific faces
        surface_faceTag = np.zeros(n_face)
        # array to store nodeID's of boundary faces
        surface_faceNodes = np.zeros(n_face * 4)

        index = 0
        # Find nodes and boundary tags
        for i in range(n_face):
            surface_faceTag[i] = self.faceInfo[[surface_faceID[i]], 0]
            for j in range(self.faceType[surface_faceID[i]]):
                index = index + 1
                surface_faceNodes[index] = self.faceNodes[4 *
                                                          surface_faceID[i] + j, 0]

        # Extract only unique nodes
        unique_nodes, unique_counts = np.unique(
            surface_faceNodes, return_counts=True)
        n_nodes = len(unique_nodes)

        f = open(fname, "w")
        f.write("TITLE = Boundary plot\n")
        f.write("VARIABLES = \"X\" \"Y\" \"Z\" \"Tag\"\n")
        f.write(
            "ZONE T=\"PURE-QUADS\", NODES={}, ELEMENTS={}, DATAPACKING=BLOCK, VARLOCATION=([4]=CELLCENTERED), ZONETYPE=FEQUADRILATERAL\n".format(n_nodes, n_face))

        # Print unique node locations
        for n in unique_nodes:
            f.write("{}\n".format(self.nodeVertex[int(n), 0]))
        for n in unique_nodes:
            f.write("{}\n".format(self.nodeVertex[int(n), 1]))
        for n in unique_nodes:
            f.write("{}\n".format(self.nodeVertex[int(n), 2]))
        for face in range(n_face):
            f.write("{}\n".format(int(surface_faceTag[face])))

        # Print nodes making up each face
        for face in range(n_face):
            for i in range(4):
                f.write("{} ".format(np.where(unique_nodes ==
                        surface_faceNodes[face * 4 + i])[0][0] + 1))
            f.write("\n")
        f.close()

        return

    def extractBoundaryNodes(self, zoneID):
        # returns a list of boundary node locations
        surface_faces = np.where(self.faceInfo[:, 0] == zoneID)[0]

        boundary_faces = np.empty((0, 3), dtype=float)

        for face in surface_faces:
            for node in range(self.faceType[face]):
                faceNodeID = self.faceIndex[face] + node
                nodeID = self.faceNodes[faceNodeID][0]
                nodeToAdd = self.nodeVertex[nodeID]
                boundary_faces = np.vstack([boundary_faces, nodeToAdd])

        return boundary_faces

    def extractHaloFaces(self):
        # Count number of halo cells in the mesh
        max_cell_ID = max(self.faceCell[:, 1])
        num_halo_cells = max_cell_ID - self.numCells
        print('num_Halo_cells= {}'.format(num_halo_cells))

        # Count the number of faces requiring a halo cell
        num_internal_faces = np.count_nonzero(self.faceBC == 0)
        num_halo_faces = self.numFaces - num_internal_faces
        print('num_halo_faces= {}'.format(num_halo_faces))

    def writeLKEconf(self, fname, r0, nbase):
        # Output .conf files for LKE mesh deformation program
        if self.V:
            print('Writing LKE mesh deformation config files')
        volfile = 'volume'
        surfile = 'surface'
        dformat = 'xyz'

        fout = open(fname, "w")
        fout.write("voltype = {} \n".format(dformat))
        fout.write("mesh = {} \n".format(volfile + '.' + dformat))
        fout.write("surfpts = {} \n".format(surfile + '.' + dformat))
        fout.write("rbfmode = 1 \n")
        fout.write("r0 = {} \n".format(r0))
        fout.write("nbase = {}".format(nbase))

        fout.close()

        fvol = open(volfile + '.' + dformat, "w")
        fvol.write("{} \n".format(self.n_v))
        for i in range(self.n_v):
            fvol.write("{:.12f} \t {:.12f} \t {:.12f} \n".format(
                self.X_v[i, 0], self.X_v[i, 1], self.X_v[i, 2]))
        fvol.close()

        fsur = open(surfile + '.' + dformat, "w")
        fsur.write("{} \n".format(self.n_s))
        for i in range(self.n_s):
            fsur.write("{:.12f} \t {:.12f} \t {:.12f} \n".format(
                self.X_s[i, 0], self.X_s[i, 1], self.X_s[i, 2]))
        fsur.close()

        if self.V:
            print('LKE config files written successfully')
        return

    def rotate_surface(self, alpha):
        self.X_def = np.zeros_like(self.X_s)

        alpha_rad = np.deg2rad(alpha)

        R = np.array([[np.cos(alpha_rad), 0, -np.sin(alpha_rad)],
                     [0, 1, 0], [np.sin(alpha_rad), 0, np.cos(alpha_rad)]])

        for i in range(self.n_s):
            self.X_def[i, :] = np.matmul(R, self.X_s[i, :]) - self.X_s[i, :]
            # self.X_def[i,0] = np.cos(alpha_rad) * self.X_s[i,0] - np.sin(alpha_rad) * self.X_s[i,2]
            # self.X_def[i,2] = np.sin(alpha_rad) * self.X_s[i,0] + np.cos(alpha_rad) * self.X_s[i,2]
        # self.X_def = self.X_s

    def translate_surface(self, vec):
        for i in range(self.n_s):
            self.X_def[i, 0] = vec[0]
            self.X_def[i, 1] = vec[1]
            self.X_def[i, 2] = vec[2]

    def generate_deformation_file(self, fname):

        fout = open(fname, "w")
        fout.write("{} \n".format(self.n_s))
        for i in range(self.n_s):
            fout.write("{:.12f} \t {:.12f} \t {:.12f} \n".format(
                self.X_def[i, 0], self.X_def[i, 1], self.X_def[i, 2]))
        fout.close()

    def rbf_rotate(self, surfID, r0, nbase, alpha):
        # Preprocessing
        self.extractSurfaceFaces(surfID)
        self.writeLKEconf('rotate.LKE', r0, nbase)

        os.system('meshprep rotate.LKE')

        self.rotate_surface(alpha)
        # vec = [0,0,5]
        # self.translate_surface(vec)

        self.generate_deformation_file('surface_deformations.xyz')

        os.system(
            'meshdef volume.xyz.meshdef surface_deformations.xyz volume_deformations.xyz')

        self.X_vol_deformed = np.loadtxt('volume_deformations.xyz', skiprows=1)

        print(self.faceType.shape, self.numFaces)
        self.nodeVertex = self.X_vol_deformed

        self.writetec('test.plt')

        self.writeh5('deformed.h5')

        os.system(
            'rm rotate.LKE surface.xyz volume.xyz surface_deformations.xyz volume_deformations.xyz volume.xyz.meshdef def.xyz')

    def get_surface_nodes(self, surfaceID, fname):
        # Find faceID's with zone tag specified
        surface_faces = np.where(self.faceInfo[:, 0] == surfaceID)[0]

        # find all the nodes making up the face
        n_surface_nodes = np.sum(self.faceType[surface_faces])
        surface_node_vertex = np.zeros([n_surface_nodes])

        index = 0
        for f in surface_faces:
            for i in range(self.faceType[f][0]):
                surface_node_vertex[index] = self.faceNodes[self.faceIndex[f] + i]
                index = index + 1

        # sort these to only get unique nodes
        unique_vertex = np.unique(surface_node_vertex)

        # get vertex points for unique nodes
        surface_nodes = np.zeros([len(unique_vertex), 3])
        for i in range(len(unique_vertex)):
            surface_nodes[i, :] = self.nodeVertex[int(unique_vertex[i]), :]

        # print surface nodes
        f = open(fname, "w")
        # f.write("{}\n".format(len(surface_nodes[:,0])))
        for nodes in surface_nodes:
            f.write("{} \t {} \t {} \n".format(nodes[0], nodes[1], nodes[2]))


class zcfd_results():
    # zCFD results data class, function needs to be called in order to create tecplot results file

    def __init__(self, fname):
        f = h5py.File(fname, "r")
        run = f.get('run')
        print(run.attrs.keys())
        self.attrs = {}
        for attr in run.attrs.keys():
            self.attrs[attr] = run.attrs.get(attr)[0]

        print(run.keys())
        self.dset = {}
        for dset in run.keys():
            self.dset[dset] = np.array(run.get(dset))

        print(self.attrs)
        print(self.dset)
        f.close()


def check_and_replace(mesh):
    ncell_nodes = np.zeros(mesh.n_cell)
    problem_cells = 1
    refinements = 1
    node_map = {}
    # pool=mp.Pool(mp.cpu_count())
    while problem_cells != 0:
        node_map = {}
        problem_cells = 0
        print(refinements)
        refinements = refinements + 1
        for c in range(mesh.ncell):
            cell_nodes = []
            faces = mesh.cellFace[6 * c: 6 * c + 6]
            for f in faces:
                cell_nodes = np.append(
                    cell_nodes, mesh.faceNodes[f * 4: f * 4 + 4])

            cell_nodes = np.unique(cell_nodes)
            num_cell_nodes = len(cell_nodes)
            ncell_nodes[c] = num_cell_nodes

            if num_cell_nodes > 8:
                problem_cells = problem_cells + 1
                for i in range(num_cell_nodes):
                    for j in range(num_cell_nodes):
                        nodei = int(cell_nodes[i])
                        nodej = int(cell_nodes[j])
                        rad = np.linalg.norm(
                            mesh.nodeVertex[nodei, :] - mesh.nodeVertex[nodej, :])
                        if rad < 0.0001 and nodei != nodej:
                            if nodei > nodej:
                                node_map[nodej] = nodei

        print(problem_cells)
        unique, counts = np.unique(ncell_nodes, return_counts=True)
        print(dict(zip(unique, counts)))

        i = 0
        for f in reversed(node_map.keys()):
            print('{} \\ {}'.format(i, len(node_map.keys())))
            mesh.sortnodes(node_map[f], f)
            i = i + 1
