import numpy as np
import h5py

""" 
Converter for CBA mesh to zCFD h5 format 

Tom Wainwright

University of Bristol 2020
tom.wainwright@bristol.ac.uk

Classes:
- CBA_mesh
    -load_cba(fname,V)
    -solve_faces()
    -structure_data()
    -get_common_faces()
    -check_face_allignment(V)
    -convert_h5_data(V)
    -check_unassigned_faces(V)
    -check_unassigned_faceNodes(V)
    -writetec(fname)

- CBA_block
    -order_points()
    -get_corners()
    -get_face_corners(face)
    -get_nfaces()
    -assign_primary_faces(cell_offset,face_offset)
    -get_boundface_ID(a,b,f)
    -assign_boundface(a,b,f,face_ID)
    -translate_BC()
    -get_faceNodes(i,j,k,p)
    
    """



class CBA_mesh():
    def __init__(self,fname='NONE',V=False):
        # Define properties of CBA mesh

        # Integers
        self.ncell = 0
        self.npts = 0
        self.nblocks = 0

        # dictionaries
        self.block = {}

        # If mesh handle is provided, load mesh
        if fname != 'NONE':
            self.load_cba(fname,V)

    def load_cba(self,fname,V=False):
        # Load CBA mesh

        data = np.loadtxt(fname)

        # Process CBA data into useful structure

        self.nblocks = int(data[0,0])

        line_index = 1

        for b in range(self.nblocks):
            # Create CBA_block class for each block
            self.block[b] = CBA_block()
            self.block[b].blockID = b

            # Load header information
            self.block[b].nptsi = int(data[line_index,0])
            self.block[b].nptsj = int(data[line_index,1])
            self.block[b].nptsk = int(data[line_index,2])

            self.block[b].npts = self.block[b].nptsi * self.block[b].nptsj * self.block[b].nptsk      
            
            line_index = line_index + 1

            # Load body information
            self.block[b].X = data[line_index:line_index + self.block[b].npts,:]
            
            line_index = line_index + self.block[b].npts

            # Load footer information
            for i in range(6):
                self.block[b].connectivity[i] = {}
                self.block[b].connectivity[i]['type']=int(data[line_index,0])
                self.block[b].connectivity[i]['neighbour']=int(data[line_index,1])
                self.block[b].connectivity[i]['orientation']=int(data[line_index,2])
                line_index = line_index + 1

            if self.block[b].nptsk == 1:
                # 2D mesh
                self.block[b].nptsk = 2
                self.block[b].X = np.concatenate((self.block[b].X,self.block[b].X))
                self.block[b].X[self.block[b].npts:,1] = 1
                self.block[b].npts = self.block[b].nptsi * self.block[b].nptsj * self.block[b].nptsk      
            
            self.block[b].ncell = (self.block[b].nptsi - 1) * (self.block[b].nptsj - 1) * (self.block[b].nptsk - 1)
            self.ncell = self.ncell + self.block[b].ncell
            self.npts = self.npts + self.block[b].npts

        if V:
            print(self.ncell)

        self.structure_data()
        self.get_common_faces()
        self.solve_faces()
        
    
    def solve_faces(self):
        self.nface = 0
        for b in range(self.nblocks):
            self.block[b].get_nfaces()
            self.nface = self.nface + self.block[b].nface

    def structure_data(self):
        # Process data into a slightly more useful format
        for b in range(self.nblocks):
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

        for b in range(self.nblocks):               # Cycle through blocks
            for i in self.block[b].connectivity:                      # Cycle through face boundaries
                if self.block[b].connectivity[i]['type'] == 2:
                    if self.block[b].connectivity[i]['neighbour'] > b+1:
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
                        
                    if self.block[b].connectivity[i]['neighbour'] < b+1:
                        n_driven = n_driven + 1

                    if self.block[b].connectivity[i]['neighbour'] == b+1:
                        print('wrap')
                        if self.block[b].connectivity[i]['orientation'] > i+1:
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

                        if self.block[b].connectivity[i]['orientation'] < i+1:
                            n_driven = n_driven + 1

        if n_driving != n_driven:
            print('ERROR- mismatch in numbers of neighbouring faces')

        self.n_commonface = n_driving

    def check_face_allignment(self,V=False):
        problem_axis = 0
        for f in range(self.n_commonface):              
            if V:
                print('MB face number: {}'.format(f))
            block1 = self.common_faces[f]['block1']
            face1 = self.common_faces[f]['face1']
            block2 = self.common_faces[f]['block2']
            face2 = self.common_faces[f]['face2']

            if V:
                print('block1: {} \tface1: {} \tblock2: {} \tface2: {}'.format(block1,face1,block2,face2))
            
            face1_corners = self.block[block1].get_face_corners(face1)
            face2_corners = self.block[block2].get_face_corners(face2)

            # Get face primary axis

            ax1 = face1_corners[0,:] - face1_corners[1,:]
            bx1 = face1_corners[0,:] - face1_corners[3,:]

            ax2 = face2_corners[0,:] - face2_corners[1,:]
            bx2 = face2_corners[0,:] - face2_corners[3,:]
            cx2 = face2_corners[3,:] - face2_corners[2,:]
            dx2 = face2_corners[1,:] - face2_corners[2,:]

            face1_axes = {'ax1': ax1, 'bx1':bx1}
            face2_axes = {'ax2': ax2, 'bx2':bx2, 'cx2':cx2, 'dx2':dx2}

            print(face1_axes)

            axis_colinear = 0
            axis_reversed = 0

            for i in face1_axes:
                for j in face2_axes:
                    cross = np.cross(face1_axes[i],face2_axes[j])
                    dot = np.dot(face1_axes[i],face2_axes[j])

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
                        if V:
                            print('Colinear: {}, {} \t direction: {}'.format(i,j,msg))

            if axis_colinear != 2:
                problem_axis = problem_axis + 1

        if problem_axis != 0:
            print('yarrr we be avin a problem')
            print(problem_axis)
        
    def convert_h5_data(self,V=False):
        cell_ID = 0
        face_ID = 0

        self.check_face_allignment(V)

        # Solve primary faces within blocks
        for b in range(self.nblocks):
            (cell_ID, face_ID) = mesh.block[b].assign_primary_faces(cell_ID,face_ID)
        if V:
            print('Cells assigned: {} \t Faces assigned: {}'.format(cell_ID,face_ID))

        # Solve block boundaries
        for f in self.common_faces:
            # Get number of points to index through
            block1 = self.common_faces[f]['block1']
            block2 = self.common_faces[f]['block2']
            face1 = self.common_faces[f]['face1']
            face2 = self.common_faces[f]['face2']

            if face2 == 0 or face2 == 1:
                npts_a = self.block[block2].nptsj
                npts_b = self.block[block2].nptsk
            elif face2 == 2 or face2 == 3:
                npts_a = self.block[block2].nptsi
                npts_b = self.block[block2].nptsk
            elif face2 == 4 or face2 == 5:
                npts_a = self.block[block2].nptsi
                npts_b = self.block[block2].nptsj

            if V:
                print('MB_faceID: {} block1: {} block2: {} face1: {} face2: {}'.format(f,block1,block2,face1,face2))
                print('npts_a: {} npts_b: {}'.format(npts_a, npts_b))

            for a in range(npts_a-1):
                for b in range(npts_b-1):
                    face_ID = self.block[block1].get_boundface_ID(a,b,face1)
                    if self.common_faces[f]['ax1']['reversed']:
                        a2 = npts_a - 2 - a
                    else:
                        a2 = a
                    if self.common_faces[f]['bx1']['reversed']:
                        b2 = npts_b - 2 - b
                    else:
                        b2 = b
                    
                    self.block[block2].assign_boundface(a2,b2,face2,face_ID)
            
        
        self.check_unassigned_faces(V)

        self.faceCell = np.ones([self.nface,2], dtype=int) * -100
        self.cellFace = np.ones([self.ncell * 6],dtype=int) * -100

        # assign faceCell dataset
        for b in range(self.nblocks):
            for i in range(self.block[b].nptsi - 1):
                for j in range(self.block[b].nptsj - 1):
                    for k in range(self.block[b].nptsk - 1):
                        for f in range(6):
                            face_ID = self.block[b].cell_face[i][j][k][f]
                            cell_ID = self.block[b].cell_face[i][j][k]['cell_ID']

                            if self.faceCell[face_ID,0] < 0:
                                self.faceCell[face_ID,0] = cell_ID
                            else:
                                self.faceCell[face_ID,1] = cell_ID

                            self.cellFace[(cell_ID * 6) + f] = face_ID

        self.check_unassigned_cellFace(V)
                            
        # assign halo cells

        nhalo = 0

        for f in range(self.nface):
            if self.faceCell[f,1] < 0:
                self.faceCell[f,1] = self.ncell + nhalo
                nhalo = nhalo + 1        
        if V:
            print('Halo cells assigned: {}'.format(nhalo))

        # assign boundary conditions

        self.faceBC = np.ones([self.nface],dtype=int) * -100
        
        nhalo_face = 0

        for b in range(self.nblocks):
            for f in self.block[b].face_BC:
                self.faceBC[f] = self.block[b].face_BC[f]
                if self.block[b].face_BC[f] != 0:
                    nhalo_face = nhalo_face + 1
        if V:
            print('Halo faces expected: {}'.format(nhalo_face))

        self.check_unassigned_faceBC(V)

        # assign faceInfo dataset
        self.faceInfo = np.zeros([self.nface,2],dtype=int)

        for b in range(self.nblocks):
            for f in self.block[b].face_info:
                self.faceInfo[f,0] = self.block[b].face_info[f]

        # assign face nodes
        self.faceNodes = np.ones([(self.nface)*4],dtype=int) * -100

        point_offset = 0

        for b in range(self.nblocks):
            for f in self.block[b].face_nodes:
                for p in range(4):
                    self.faceNodes[f*4 + p] = int(self.block[b].face_nodes[f][p] + point_offset)
            point_offset = point_offset + self.block[b].npts

        self.check_unassigned_faceNodes(V)
        
        # create nodeVertex dataset
        for b in range(self.nblocks):
            if b == 0:
                self.nodeVertex = self.block[b].X
            else:
                self.nodeVertex = np.concatenate((self.nodeVertex,self.block[b].X))

        # assign faceType
        self.faceType = np.zeros([self.nface,2],dtype=int)
        for f in range(self.nface):
            self.faceType[f,0] = 4
        
        self.remove_common_nodes(V)
            
    def check_unassigned_faces(self,V=False):
        if V:
            print('Checking for unassigned faces...')
        unassigned_faces = 0
        for b in range(self.nblocks):
            for i in range(self.block[b].nptsi - 1):
                for j in range(self.block[b].nptsj - 1):
                    for k in range(self.block[b].nptsk - 1):
                        for f in range(6):
                            if self.block[b].cell_face[i][j][k][f] == 'hold':
                                unassigned_faces = unassigned_faces + 1    
        if V:
            print('{} faces unassigned'.format(unassigned_faces))

    def check_unassigned_faceNodes(self,V=False):
        if V:
            print('Checking for unassigned faceNodes...')
        unassigned_faceNodes = 0
        for f in self.faceNodes:
            if f < 0:
                unassigned_faceNodes = unassigned_faceNodes + 1            
        if V:
            print('{} faceNodes unassigned'.format(unassigned_faceNodes))

    def check_unassigned_faceBC(self,V=False):
        if V:
            print('Checking for unassigned faceBC...')
        unassigned_faceBC = 0
        for f in self.faceBC:
            if f < 0:
                unassigned_faceBC = unassigned_faceBC + 1            
        if V:
            print('{} faceBC unassigned'.format(unassigned_faceBC))
 
    def check_unassigned_cellFace(self,V=False):
        if V:
            print('Checking for unassigned cellFaces')
        unassigned_cellFace = 0
        for f in self.cellFace:
            if f < 0:
                unassigned_cellFace = unassigned_cellFace + 1
        if V: 
            print('{} cellFaces unassigned'.format(unassigned_cellFace))

    def check_cell_references(self,V=False):
        if V:
            print('Checking cell references')
        cell_references = np.zeros(self.ncell,dtype=int)
        for f in range(self.nface):
            cell_references[self.faceCell[f,0]] = cell_references[self.faceCell[f,0]] + 1
            if self.faceCell[f,1] < self.ncell:
                cell_references[self.faceCell[f,1]] = cell_references[self.faceCell[f,1]] + 1

    def remove_common_nodes(self,V=False):
        if V:
            print('Removing common nodes')

        node_references = np.zeros_like(self.nodeVertex[:,0])
        presort_nodes = len(self.nodeVertex[:,0])

        for fn in self.faceNodes:
            node_references[fn] = node_references[fn] + 1

        unique, counts = np.unique(node_references, return_counts=True)

        if V:
            print('Number of node references (pre-sort)')
            print(dict(zip(unique, counts)))

        unique, indices = np.unique(self.nodeVertex, axis = 0, return_inverse=True)
                
        faceNodes_sorted = np.zeros_like(self.faceNodes)

        for fn in range(len(self.faceNodes)):
            faceNodes_sorted[fn] = indices[self.faceNodes[fn]]

        self.faceNodes_sorted = faceNodes_sorted
        self.nodeVertex_sorted = unique
        postsort_nodes = len(self.nodeVertex_sorted[:,0])

        node_references = np.zeros_like(self.nodeVertex_sorted[:,0])

        for fn in self.faceNodes_sorted:
            node_references[fn] = node_references[fn] + 1

        unique, counts = np.unique(node_references, return_counts=True)
        if V:
            print('Number of node references (post-sort)')
            print(dict(zip(unique, counts)))

            print('{} Common nodes removed'.format(presort_nodes - postsort_nodes))
        



    def writetec(self,fname,V=True):
        # Write ZCFD mesh to tecplot FEPOLYHEDRON FORMAT
        if V:
            print('Writing tecplot mesh file: {}'.format(fname))
        n_v = np.size(self.nodeVertex_sorted[:,0])
        n_c = self.ncell
        n_f = self.nface
        n_fnodes = np.size(self.faceNodes_sorted) 

        fout = open(fname,"w")
        if V:
            print('Writing Header Information')
        fout.write("VARIABLES= \"X\" \"Y\" \"Z\"\n")
        fout.write("ZONE \n")
        fout.write("NODES = {} \n".format(n_v))                     # Number of Nodes
        fout.write("FACES = {} \n".format(n_f))                     # Number of faces
        fout.write("TOTALNUMFACENODES = {} \n".format(n_fnodes))    # Number of nodes in faces
        fout.write("NUMCONNECTEDBOUNDARYFACES = 0 \n")              # Number of connected boundary faces (0)
        fout.write("TOTALNUMBOUNDARYCONNECTIONS = 0 \n")            # Number of connected zones (0)
        fout.write("ELEMENTS = {} \n".format(n_c))                  # Number of cells
        fout.write("DATAPACKING = BLOCK \n")                        # Data formatting- must be block for FEPOLYHEDRON
        fout.write("ZONETYPE = FEPOLYHEDRON \n")                    # Mesh type- FE polyhedron for zCFD

        if V:
            print('Writing Node Vertex Points')
        fout.write('# i Vertex Locations \n')
        for i in range(n_v):
            fout.write("{} \n".format(self.nodeVertex_sorted[i,0]))
        fout.write('# j Vertex Locations \n')
        for i in range(n_v):
            fout.write("{} \n".format(self.nodeVertex_sorted[i,1]))
        fout.write('# k Vertex Locations \n')
        for i in range(n_v):
            fout.write("{} \n".format(self.nodeVertex_sorted[i,2]))

        if V:
            print('Writing Face Info')
        fout.write('# Number of points per face \n')
        for i in range(n_f):
            fout.write("{} \n".format(self.faceType[i,0]))

        if V:
            print('Writing Face Nodes')
        fout.write('# Nodes making up each face \n')
        for i in range(n_f):
            n_points = int(self.faceType[i,0])
            for j in range(n_points):
                index = i * n_points + j
                fout.write("{} ".format(self.faceNodes_sorted[index]+1))
            fout.write("\n")

        if V:
            print('Writing Face Cell Interfaces')
        fout.write('# Left Cells \n')
        for i in range(n_f):
            fout.write("{} \n".format(int(self.faceCell[i,0]+1)))
        fout.write('# Right Cells \n')
        for i in range(n_f):
            if self.faceCell[i,1] < n_c:
                fout.write("{} \n".format(int(self.faceCell[i,1]+1)))
            elif self.faceCell[i,1] >= n_c:
                fout.write("0 \n")
        
        if V:
            print('tecplot file written successfully')

    def write_h5(self,fname):
        f = h5py.File(fname,"w")
        h5mesh = f.create_group("mesh")

        h5mesh.attrs.create("numFaces",self.nface, shape=(1,1))
        h5mesh.attrs.create("numCells",self.ncell, shape=(1,1,))

        h5mesh.create_dataset("cellFace", data=self.cellFace)
        h5mesh.create_dataset("faceBC", data=self.faceBC, shape=(self.nface,1))
        h5mesh.create_dataset("faceCell", data=self.faceCell)
        h5mesh.create_dataset("faceInfo", data=self.faceInfo)
        h5mesh.create_dataset("faceNodes", data=self.faceNodes, shape=(self.nface * 4,1))
        h5mesh.create_dataset("faceType", data=self.faceType[:,0], shape=(self.nface,1))
        h5mesh.create_dataset("nodeVertex", data=self.nodeVertex)


class CBA_block():
    def __init__(self):
        # Define properties of CBA block

        # Integers

        self.ncell = 0
        self.npts = 0
        self.nptsi = 0
        self.nptsj = 0
        self.nptsk = 0
        self.blockID = 0
        
        # Arrays
        self.X = []

        # Dictionaries
        self.connectivity = {}

    def order_points(self):
        # Re-structure points to 3D array
        self.pts = np.zeros([self.nptsi,self.nptsj,self.nptsk,3])
 
        
        index = 0

        for k in range(self.nptsk):
            for j in range(self.nptsj):
                for i in range(self.nptsi):
                    self.pts[i,j,k,0] = self.X[index,0]
                    self.pts[i,j,k,1] = self.X[index,1]
                    self.pts[i,j,k,2] = self.X[index,2]
                    index = index + 1

    def get_corners(self):
        corners = np.zeros([8,3])
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
                    corners[index,0] = self.pts[i*(self.nptsi-1),j*(self.nptsj-1),k*(self.nptsk-1),0]
                    corners[index,1] = self.pts[i*(self.nptsi-1),j*(self.nptsj-1),k*(self.nptsk-1),1]
                    corners[index,2] = self.pts[i*(self.nptsi-1),j*(self.nptsj-1),k*(self.nptsk-1),2]
                    
                    index = index + 1

        return(corners)
    
    def get_face_corners(self,face):

        corners = self.get_corners()

        face_corners = np.zeros([4,3])

        if face == 0:       # i min
            f_V = [0,2,6,4]
        elif face == 1:     # i max
            f_V = [1,3,7,5]
        elif face == 2:     # j min
            f_V = [0,1,5,4]
        elif face == 3:     # j max
            f_V = [2,3,7,6]
        elif face == 4:     # k min
            f_V = [0,1,3,2]
        elif face == 5:     # k max
            f_V = [4,5,7,6]

        for i in range(4):
            face_corners[i,:] = corners[f_V[i],:]

        return(face_corners)

    def get_nfaces(self):

        # Number of faces in plane (jk is i face)
        nface_jk = (self.nptsj - 1) * (self.nptsk - 1)
        nface_ik = (self.nptsi - 1) * (self.nptsk - 1)
        nface_ij = (self.nptsi - 1) * (self.nptsj - 1)

        i_planes = self.nptsi
        j_planes = self.nptsj
        k_planes = self.nptsk

        for f in self.connectivity:
            if self.connectivity[f]['type'] == 2:
                if not self.connectivity[f]['driving']:
                    if f == 0 or f == 1:
                        i_planes = i_planes - 1
                    elif f == 2 or f == 3:
                        j_planes = j_planes - 1
                    elif f == 4 or f == 5:
                        k_planes = k_planes - 1

        self.nface_i = nface_jk * i_planes
        self.nface_j = nface_ik * j_planes
        self.nface_k = nface_ij * k_planes

        self.nface = self.nface_i + self.nface_j + self.nface_k

        self.nbound_face = (nface_jk + nface_ik + nface_ij) * 2

    
    def assign_primary_faces(self,cell_offset,face_offset):
        self.cell_face = {}                       # Faces on each cell
        self.face_BC = {}
        self.face_nodes = {}
        self.face_info = {}

        cell_ID = 0
        face_ID = 0
        boundary_faces = 0
        iface_ID = 0

        boundary_conditions = [0,self.nptsi-2,0,self.nptsj-2,0,self.nptsk-2]
        internal_conditions = [False,True,False,True,False,True]
        facenode_conditions =  {0: {0:[0,2,4],1:[0,2,5],2:[0,3,5],3:[0,3,4]},
                                1: {0:[1,2,4],1:[1,3,4],2:[1,3,5],3:[1,2,5]},
                                2: {0:[0,2,4],1:[1,2,4],2:[1,2,5],3:[0,2,5]},
                                3: {0:[0,3,4],1:[0,3,5],2:[1,3,5],3:[1,3,4]},
                                4: {0:[0,2,4],1:[0,3,4],2:[1,3,4],3:[1,2,4]},
                                5: {0:[0,2,5],1:[1,2,5],2:[1,3,5],3:[0,3,5]}}
        

        for i in range(self.nptsi-1):
            self.cell_face[i] = {}
            for j in range(self.nptsj-1):
                self.cell_face[i][j] = {}
                for k in range(self.nptsk-1):

                    # Number cells
                    self.cell_face[i][j][k] = {}
                    self.cell_face[i][j][k]['cell_ID'] = cell_ID + cell_offset

                    # Number faces

                    # identify if at boundary- then if unique face
                    position = [i,i,j,j,k,k]
                    corners = [i,i+1,j,j+1,k,k+1]

                    for p in range(6):
                        if position[p] == boundary_conditions[p]:
                            boundary_faces = boundary_faces + 1
                            if self.connectivity[p]['type'] == 2:
                                if not self.connectivity[p]['driving']:
                                    self.cell_face[i][j][k][p] = 'hold'
                                else:
                                    self.cell_face[i][j][k][p] = face_ID + face_offset
                                    self.face_BC[face_ID + face_offset] = 0    # Internal boundary face
                                    self.face_info[face_ID + face_offset] = 0
                                    self.face_nodes[face_ID + face_offset] = self.get_faceNodes(i,j,k,p)

                                    face_ID = face_ID + 1

                            else:
                                self.cell_face[i][j][k][p] = face_ID + face_offset
                                self.face_BC[face_ID + face_offset] = self.connectivity[p]['BC_translated']
                                self.face_info[face_ID + face_offset] = self.connectivity[p]['FT_translated']
                                self.face_nodes[face_ID + face_offset] = self.get_faceNodes(i,j,k,p)
                                
                                face_ID = face_ID + 1


                        elif internal_conditions[p]:
                            self.cell_face[i][j][k][p] = face_ID + face_offset
                            self.face_BC[face_ID + face_offset] = 0
                            self.face_info[face_ID + face_offset] = self.connectivity[p]['FT_translated']
                            self.face_nodes[face_ID + face_offset] = self.get_faceNodes(i,j,k,p)
                            
                            face_ID = face_ID + 1

                        elif p == 0:
                            self.cell_face[i][j][k][0] = self.cell_face[i-1][j][k][1]
                        elif p == 2:
                            self.cell_face[i][j][k][2] = self.cell_face[i][j-1][k][3]
                        elif p == 4:
                            self.cell_face[i][j][k][4] = self.cell_face[i][j][k-1][5]

                    cell_ID = cell_ID + 1

        if face_ID != self.nface:
            print('Mismatch in face numbers: {} assigned, {} expected'.format(face_ID,self.nface))
            print('Difference of {}'.format(self.nface - face_ID))
        if cell_ID != self.ncell:
            print('Mismatch in cell numbers: {} assigned, {} expected'.format(cell_ID,self.ncell))
            print('Difference of {}'.format(self.ncell - cell_ID))

        return (cell_ID + cell_offset), (face_ID + face_offset)
    
    def get_boundface_ID(self,a,b,f):
        face_ID = 0

        # print(a,b,f,self.blockID)
        if f == 0:
            face_ID = self.cell_face[0][a][b][0]
        elif f == 1:
            face_ID = self.cell_face[self.nptsi-2][a][b][1]
        elif f == 2:
            face_ID = self.cell_face[a][0][b][2]
        elif f == 3:
            face_ID = self.cell_face[a][self.nptsj-2][b][3]
        elif f == 4:
            face_ID = self.cell_face[a][b][0][4]
        elif f == 5:
            face_ID = self.cell_face[a][b][self.nptsk-2][5]

        return face_ID

    def assign_boundface(self,a,b,f,face_ID):
        # Check we're not pushing a face to a non-driven face
        if self.connectivity[f]['driving']:
            print('ERROR- PUSHING TO NON DRIVEN FACE')
            print('Pushing to block: {} face: {}'.format(self.blockID,f))
        if f == 0:
            self.cell_face[0][a][b][0] = face_ID
        elif f == 1:
            self.cell_face[self.nptsi-2][a][b][1] = face_ID
        elif f == 2:
            self.cell_face[a][0][b][2] = face_ID
        elif f == 3:
            self.cell_face[a][self.nptsj-2][b][3] = face_ID
        elif f == 4:
            self.cell_face[a][b][0][4] = face_ID
        elif f == 5:
            self.cell_face[a][b][self.nptsk-2][5] = face_ID

    def translate_BC(self):
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

        BC_dict = {-2:7, -1:3, 0:3, 1:9, 2:0, 3:12, 4:12}
        FT_dict = {-2:7, -1:4, 0:3, 1:2, 2:0, 3:5, 4:6}
        for f in range(6):
            self.connectivity[f]['BC_translated'] = BC_dict[self.connectivity[f]['type']]
            self.connectivity[f]['FT_translated'] = FT_dict[self.connectivity[f]['type']]

    def get_faceNodes(self,i,j,k,p):
        if p == 0:
            nv1 = i + j*self.nptsi + k*self.nptsi*self.nptsj
            nv2 = i + j*self.nptsi + (k+1)*self.nptsi*self.nptsj
            nv3 = i + (j+1)*self.nptsi + (k+1)*self.nptsi*self.nptsj
            nv4 = i + (j+1)*self.nptsi + k*self.nptsi*self.nptsj
        elif p == 1:
            nv1 = (i+1) + j*self.nptsi + k*self.nptsi*self.nptsj
            nv2 = (i+1) + (j+1)*self.nptsi + k*self.nptsi*self.nptsj
            nv3 = (i+1) + (j+1)*self.nptsi + (k+1)*self.nptsi*self.nptsj
            nv4 = (i+1) + j*self.nptsi + (k+1)*self.nptsi*self.nptsj
        elif p == 2:
            nv1 = i + j*self.nptsi + k*self.nptsi*self.nptsj
            nv2 = (i+1) + j*self.nptsi + k*self.nptsi*self.nptsj
            nv3 = (i+1) + j*self.nptsi + (k+1)*self.nptsi*self.nptsj
            nv4 = i + j*self.nptsi + (k+1)*self.nptsi*self.nptsj
        elif p == 3:
            nv1 = i + (j+1)*self.nptsi + k*self.nptsi*self.nptsj
            nv2 = i + (j+1)*self.nptsi + (k+1)*self.nptsi*self.nptsj
            nv3 = (i+1) + (j+1)*self.nptsi + (k+1)*self.nptsi*self.nptsj
            nv4 = (i+1) + (j+1)*self.nptsi + k*self.nptsi*self.nptsj
        elif p == 4:
            nv1 = i + j*self.nptsi + k*self.nptsi*self.nptsj
            nv2 = i + (j+1)*self.nptsi + k*self.nptsi*self.nptsj
            nv3 = (i+1) + (j+1)*self.nptsi + k*self.nptsi*self.nptsj
            nv4 = (i+1) + j*self.nptsi + k*self.nptsi*self.nptsj
        elif p == 5:
            nv1 = i + j*self.nptsi + (k+1)*self.nptsi*self.nptsj
            nv2 = (i+1) + j*self.nptsi + (k+1)*self.nptsi*self.nptsj
            nv3 = (i+1) + (j+1)*self.nptsi + (k+1)*self.nptsi*self.nptsj
            nv4 = i + (j+1)*self.nptsi + (k+1)*self.nptsi*self.nptsj

        return [nv4,nv3,nv2,nv1]


        
# mesh = CBA_mesh(fname='../../data/3D/IEA_15MW/IEA_15MW_500K.blk')
# mesh = CBA_mesh(fname='../../data/3D/MDO/MDO_250K.blk')
mesh = CBA_mesh(fname='../../data/2D/2D_test/Omesh.blk')
# mesh = CBA_mesh('../../data/3D/CT_rotor/CT0_250K.blk')
mesh.convert_h5_data(V=True)
# mesh.write_h5('../../data/3D/MDO/MDO_250K.blk.h5')

mesh.write_h5('../../data/2D/2D_test/Omesh.test.h5')
mesh.writetec('test.plt')

