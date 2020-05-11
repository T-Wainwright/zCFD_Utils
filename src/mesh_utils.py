import numpy as np 
import h5py
import sys

""" 
Utilities for handling CBA and ZCFD meshes

Tom Wainwright

University of Bristol 2020

tom.wainwright@bristol.ac.uk


Classes: 
-   cba_mesh: CBA format multiblock structured mesh
    -   load_cba(fname)
    -   h5_conv()
    -   writetec(fname)
    -   writeblk(fname)
    -   show_connectivity()

-   cba_block: single block in CBA mesh
    -   process_data()
    -   conv_h5_data(f_offset,c_offset,p_offset,ghostID,h5)

-   h5_mesh: Unstructured h5 format mesh
    -   load_zcfd(fname)
    -   writetec(fname)
    -   writeh5(fname)
    -   extractSurfaceFaces(facetag)
    
    In progress
    -   applyDeformation(def_func)
    -   writeLKE()
    -   reconstruct()

"""
class cba_mesh:
    def __init__(self,fname='NONE',V=False):
        self.n_blocks = 0               # Number of blocks
        self.npts = 0                   # Number of points total in the mesh
        self.block = {}                 # Number of points per block
        self.data_path = '../../data/'     # Path to data
        self.nface = 0
        self.ncell = 0

        self.Common_faces = {}

        self.f_com = 0

        self.V = V
        
        if fname != 'NONE':
            fname = self.data_path + fname
            self.load_cba(fname,V)

    def load_cba(self,fname,V):
        # Load CBA structured multiblock mesh
        print('Loading CBA file')
        DATA = np.loadtxt(fname,dtype='f')
        self.n_blocks = int(DATA[0,0])
        row = 1
        

        for i in range(self.n_blocks):
            self.block[i] = cba_block()
            self.block[i].blockID = i
            self.block[i].npts_i = int(DATA[row,0])
            self.block[i].npts_j = int(DATA[row,1])
            self.block[i].npts_k = int(DATA[row,2])

            # Load individual block data
            if self.block[i].npts_k == 1:
                # 2D Block
                print('2D block detected, adding duplicate plane')
                self.block[i].npts = self.block[i].npts_i * self.block[i].npts_j * self.block[i].npts_k
                self.block[i].X = DATA[row+1:self.block[i].npts+row+1, :]

                # Add Duplicate plane
                dup_plane = np.transpose(np.array([self.block[i].X[:,0], np.ones_like(self.block[i].X[:,0]), self.block[i].X[:,2]]))
                self.block[i].X = np.append(self.block[i].X,dup_plane,axis=0)
                self.block[i].connectivity = DATA[self.block[i].npts + row + 1:self.block[i].npts+row+7 ,:]
                row = row + self.block[i].npts + 7

                self.block[i].npts_k = 2
                self.block[i].npts = self.block[i].npts_i * self.block[i].npts_j * self.block[i].npts_k
            else:
                print('3D block detected')
                # 3D block
                self.block[i].npts = self.block[i].npts_i * self.block[i].npts_j * self.block[i].npts_k
                self.block[i].X = DATA[row+1:self.block[i].npts+row+1,:]
                self.block[i].connectivity = DATA[self.block[i].npts + row + 1:self.block[i].npts+row+7, :]
                row = row + self.block[i].npts + 7
            
            # Obtain shared faces dictionary
            for j in range(6):
                if self.block[i].connectivity[j,0] == 2:        # Is it an internal face
                    self.Common_faces[self.f_com] = {'block1' : i, 'block2' : int(self.block[i].connectivity[j,1]-1), 'face1' : j, 'face2' : int(self.block[i].connectivity[j,2]-1)}
                    self.f_com = self.f_com + 1

        # Cycle through common faces, ensuring there is only 1 entry per pair
        for i in range(self.f_com):
            # Case 1- common faces are from same block- 'Wrap around'
            if self.Common_faces[i]['block1'] == self.Common_faces[i]['block2']:
                print('wrap')
                # Pair up faces to be wrapped in ascending order
                if self.Common_faces[i]['face1'] > self.Common_faces[i]['face2']:
                    del self.Common_faces[i] 
            # Case 2- common faces are from different blocks- stitch
            else:
                print('stitch')
                if self.Common_faces[i]['block1'] > self.Common_faces[i]['block2']:
                    del self.Common_faces[i]

        for i in self.Common_faces:
            block1 = self.Common_faces[i]['block1']
            block2 = self.Common_faces[i]['block2']
            face1 = self.Common_faces[i]['face1']
            face2 = self.Common_faces[i]['face2']

            self.block[block2].falsefaces = np.append(self.block[block2].falsefaces,face2)
            self.block[block2].common_faces[face2] = {'block1' : block1, 'face1' : face1}
            if face1 == face2:
                self.block[block2].common_faces[face2]['type'] = 'mirror'
            else:
                self.block[block2].common_faces[face2]['type'] = 'alligned'
        
        for i in range(self.n_blocks):
            self.block[i].process_data()
            self.nface = self.nface + self.block[i].nface 
            self.ncell = self.ncell + self.block[i].ncell 

        if self.V:
            print('Successfully loaded CBA mesh')
            print('n_blocks = {} \t n_cells = {}'.format(self.n_blocks,self.ncell))
            for i in self.block:
                print('block = {} \t n_cells = {} \t n_i = {} \t n_j = {} \t n_k = {}'.format(self.block[i].blockID,self.block[i].ncell,self.block[i].npts_i,self.block[i].npts_j,self.block[i].npts_k))
            print('Common_faces = {}'.format(self.f_com))
            for i in self.Common_faces:
                print('block1 = {} \t face1 = {} \t block2 = {} \t face2 = {}'.format(self.Common_faces[i]['block1'],self.Common_faces[i]['face1'],self.Common_faces[i]['block2'],self.Common_faces[i]['face2']))
      
    def h5_conv(self):
        # Convert CBA format multiblock mesh to zcfd h5 format
        if self.V:
            print('Converting CBA to H5 format')
        h5 = h5_mesh(ncell=self.ncell, nface=self.nface)
        f_offset = 0
        c_offset = 0
        p_offset = 0
        ghostID = self.ncell
        for i in self.block:
            self.block[i].conv_h5_data(f_offset,c_offset,p_offset,ghostID,h5)
            f_offset = f_offset + self.block[i].nface
            c_offset = c_offset + self.block[i].ncell
            p_offset = p_offset + self.block[i].npts
            ghostID = ghostID + self.block[i].ghostID
        return(h5)
    
    def writetec(self,fname):
        # Convert CBA mesh to tecplot visualisation format
        if self.V:
            print('Generating Structured tecplot file')
        fout = open(fname,"w")
        fout.write('VARIABLES = "X" "Y" "Z" \n')
        for i in self.block:
            fout.write('ZONE I= {} J= {} K= {} F=POINT \n'.format(self.block[i].npts_i,self.block[i].npts_j,self.block[i].npts_k))
            for j in range(self.block[i].npts):
                fout.write('{:.15f} \t {:.15f} \t {:.15f} \n'.format(self.block[i].X[j,0],self.block[i].X[j,1],self.block[i].X[j,2]))
        return

    def writeblk(self,fname):
        # Write saved mesh back to CBA format
        if self.V:
            print('Generating blk mesh file')
        fout = open(fname,"w")
        fout.write('{} \t {} \t {}\n'.format(self.n_blocks,1,2.5))
        for i in self.block:
            fout.write('{} \t {} \t {}\n'.format(self.block[i].npts_i,self.block[i].npts_j,self.block[i].npts_k))
            for j in range(self.block[i].npts):
                fout.write('{:.15f} \t {:.15f} \t {:.15f} \n'.format(self.block[i].X[j,0],self.block[i].X[j,1],self.block[i].X[j,2]))
            for j in range(6):
                fout.write('{} \t {} \t {} \n'.format(int(self.block[i].connectivity[j,0]),int(self.block[i].connectivity[j,1]),int(self.block[i].connectivity[j,2])))
        return

    def show_connectivity(self):
        # Show block connectivity
        for i in self.block:
            print('{} \t'.format(i))
            for j in range(6):
                print('{} {} {} \t'.format(int(self.block[i].connectivity[j,0]),int(self.block[i].connectivity[j,1]),int(self.block[i].connectivity[j,2])))
            print('\n')
        return

class cba_block:
    def __init__(self):
        # Integers
        self.blockID = 0
        self.npts = 0
        self.npts_i = 0
        self.npts_j = 0
        self.npts_k = 0

        self.nface = 0
        self.nface_i = 0
        self.nface_j = 0
        self.nface_k = 0

        self.ncell = 0

        # Arrays
        self.X = np.array((0,3))
        self.falsefaces = []
        self.falsefaces_log = [False,False,False,False,False,False]

        # Dictionaries
        self.common_faces = {0:{},1:{},2:{},3:{},4:{},5:{}}
        self.face_bounds = {}
        self.cell_bounds = {}

        # Block connectivity
        self.connectivity = {}
        # imintag   neighbour   orientation
        # imaxtag   neighbour   orientation
        # jmintag   neighbour   orientation
        # jmaxtag   neighbour   orientation
        # kmintag   neighbour   orientation
        # kmaxtag   neighbour   orientation

        # -1 = Solid surface (Include in loads calc)
        # 0 = Solid surface
        # 1 = Farfield
        # 2 = Internal Face         Only these require neighbour and orientation flags
        # 3 = Periodic Downstream   Only these require neighbour and orientation flags
        # 4 = Periodic Upstream     Only these require neighbour and orientation flags

        # Orientation flags:

        # 1 = Connected to i min face
        # 2 = Connected to i max face
        # 3 = Connected to j min face
        # 4 = Connected to j max face
        # 5 = Connected to k min face
        # 6 = Connected to k max face
        

    def process_data(self):
        # Process individual block properties
        print('Assigning block data')
        

        # Number of points for each plane
        self.jk_pts = (self.npts_j - 1) * (self.npts_k - 1)
        self.ik_pts = (self.npts_i - 1) * (self.npts_k - 1)
        self.ij_pts = (self.npts_i - 1) * (self.npts_j - 1)

        # Calculate number of false planes
        false_i = 0
        false_j = 0 
        false_k = 0

        for f in self.falsefaces:
            f = int(f)
            # self.BC_condition[f] = self.BC_condition[f] + f_action[f]
            self.falsefaces_log[f] = True

            if f == 0 or f == 1:
                false_i = false_i + 1
            elif f == 2 or f == 3:
                false_j = false_j + 1
            elif f == 4 or f == 5:
                false_k = false_k + 1
        
        self.nface_i = (self.npts_i-false_i) * self.jk_pts
        self.nface_j = (self.npts_j-false_j) * self.ik_pts
        self.nface_k = (self.npts_k-false_k) * self.ij_pts

        self.nface = self.nface_i + self.nface_j + self.nface_k
        self.ncell = (self.npts_i - 1) * (self.npts_j - 1) * (self.npts_k - 1)
        return
    
    def conv_h5_data(self,f_offset,c_offset,p_offset,ghostID,h5):
        # Convert block data into h5 format
        print('Calculating h5 data')

        cellID = 0

        # Translate boundary conditions
        BC_trans = np.zeros(6)
        for n in range(6):                                                      # Translate boundary condition flags
            if self.connectivity[n,0] == -1 or self.connectivity[n,0] == 0:
                BC_trans[n] = 3
            elif self.connectivity[n,0] == 1:
                BC_trans[n] = 9
            elif self.connectivity[n,0] == 2:
                BC_trans[n] = 0
            elif self.connectivity[n,0] == 3 or self.connectivity[n,0] == 4:
                BC_trans[n] = 0           
            self.face_bounds[n] = []
            self.cell_bounds[n] = []
        self.BC_condition = [0,self.npts_i-1,0,self.npts_j-1,0,self.npts_k-1]
        
        # Check for intenal / periodc faces. Keep a record of these faces, along with adjoining blocks
        false_faces_i = 0
        false_faces_j = 0
        false_faces_k = 0

        cellFace = np.zeros((self.ncell,6))
        for k in range(self.npts_k-1):
            for j in range(self.npts_j-1):
                for i in range(self.npts_i-1):
                    i_1 = f_offset + i + (j*self.npts_i) + (k*(self.npts_i)*(self.npts_j-1)) - false_faces_i
                    i_2 = i_1 + 1


                    j_1 = f_offset + self.nface_i + i + (j*(self.npts_i-1)) + (k*(self.npts_i-1)*self.npts_j) - false_faces_j
                    j_2 = j_1 + (self.npts_i-1)


                    k_1 = f_offset + self.nface_i + self.nface_j + i + (j*(self.npts_i-1)) + (k*(self.npts_i-1)*(self.npts_j-1)) - false_faces_k
                    k_2 = k_1 + (self.npts_i-1)*(self.npts_j-1)

                    # Examing boundary conditions
                    face = [i,i+1,j,j+1,k,k+1]
                    jk_index = j + k*(self.npts_j - 1)
                    ik_index = i + k*(self.npts_i - 1)
                    ij_index = i * j*(self.npts_i - 1)
                    f_index = [jk_index,jk_index,ik_index,ik_index,ij_index,ij_index]
                    cellFace[cellID,:] = [i_1,i_2,j_1,j_2,k_1,k_2]

                    for f in range(6):
                        if face[f] == self.BC_condition[f]:
                            self.face_bounds[f] = np.append(self.face_bounds[f],cellFace[cellID,f])
                            self.cell_bounds[f] = np.append(self.cell_bounds[f],cellID+c_offset)
                            if self.connectivity[f,0] == 2:
                                if self.falsefaces_log[f]:
                                    # get faceID from source
                                    block1 = self.common_faces[f]['block1']
                                    face1 = self.common_faces[f]['face1']
                                    ftype = self.common_faces[f]['type']
                                    if ftype == 'alligned':
                                        # print('alligned')
                                        cellFace[cellID,f] = cba_mesh.block[block1].face_bounds[face1][f_index[f]]
                                    elif ftype == 'mirror':
                                        # print('mirror')
                                        cellFace[cellID,f] = cba_mesh.block[block1].face_bounds[face1][(self.npts_i - i -2)]
                                    # print('driven')
                                    if f == 0 or f == 1:
                                        false_faces_i = false_faces_i + 1
                                    elif f == 2 or f == 3:
                                        false_faces_j = false_faces_j + 1
                                    elif f == 4 or f == 5:
                                        false_faces_k = false_faces_k + 1
                                    h5.faceCell[int(cellFace[cellID,f]),0] = cba_mesh.block[block1].cell_bounds[face1][f_index[f]]
                                    h5.faceCell[int(cellFace[cellID,f]),1] = cellID
                            elif self.connectivity[f,0] == -2 or self.connectivity[f,0] == -1 or self.connectivity[f,0] == 0 or self.connectivity[f,0] == 1:
                                # Add ghost cells- always ghost cell OUTSIDE
                                if f == 0 or f == 1:
                                    h5.faceCell[int(cellFace[cellID,f]),0] = cellID + c_offset
                                    h5.faceCell[int(cellFace[cellID,f]),1] = ghostID
                                    ghostID = ghostID + 1
                                elif f == 2 or f == 3:
                                    h5.faceCell[int(cellFace[cellID,f]),0] = cellID + c_offset
                                    h5.faceCell[int(cellFace[cellID,f]),1] = ghostID
                                    ghostID = ghostID + 1                                   
                                elif f == 4 or f == 5:
                                    h5.faceCell[int(cellFace[cellID,f]),0] = cellID + c_offset
                                    h5.faceCell[int(cellFace[cellID,f]),1] = ghostID
                                    ghostID = ghostID + 1
                            h5.faceBC[int(cellFace[cellID,f])] = BC_trans[f]
                        else:
                            if cellID > h5.faceCell[int(cellFace[cellID,f]),1]:
                                if f == 0 or f == 1:
                                    h5.faceCell[int(cellFace[cellID,f]),0] = h5.faceCell[int(cellFace[cellID,f]),1]
                                    h5.faceCell[int(cellFace[cellID,f]),1] = cellID + c_offset
                                elif f == 2 or f == 3:
                                    h5.faceCell[int(cellFace[cellID,f]),0] = h5.faceCell[int(cellFace[cellID,f]),1]
                                    h5.faceCell[int(cellFace[cellID,f]),1] = cellID + c_offset
                                elif f == 4 or f == 5:
                                    h5.faceCell[int(cellFace[cellID,f]),0] = h5.faceCell[int(cellFace[cellID,f]),1]
                                    h5.faceCell[int(cellFace[cellID,f]),1] = cellID + c_offset

                    i_1 = int(cellFace[cellID,0])
                    i_2 = int(cellFace[cellID,1])
                    j_1 = int(cellFace[cellID,2])
                    j_2 = int(cellFace[cellID,3])
                    k_1 = int(cellFace[cellID,4])
                    k_2 = int(cellFace[cellID,5])
                                    
                    # Print off face nodes - could get negative volumes if not careful here
                    h5.faceNodes[(i_1*4)+0] = p_offset + i + (j)*self.npts_i + (k)*self.npts_i*self.npts_j
                    h5.faceNodes[(i_1*4)+1] = p_offset + i + (j)*self.npts_i + (k+1)*self.npts_i*self.npts_j
                    h5.faceNodes[(i_1*4)+2] = p_offset + i + (j+1)*self.npts_i + (k+1)*self.npts_i*self.npts_j
                    h5.faceNodes[(i_1*4)+3] = p_offset + i + (j+1)*self.npts_i + (k)*self.npts_i*self.npts_j
              
                    h5.faceNodes[(j_1*4)+0] = p_offset + i + (j)*self.npts_i + (k)*self.npts_i*self.npts_j
                    h5.faceNodes[(j_1*4)+1] = p_offset + (i+1) + (j)*self.npts_i + (k)*self.npts_i*self.npts_j
                    h5.faceNodes[(j_1*4)+2] = p_offset + (i+1) + (j)*self.npts_i + (k+1)*self.npts_i*self.npts_j
                    h5.faceNodes[(j_1*4)+3] = p_offset + (i) + (j)*self.npts_i + (k+1)*self.npts_i*self.npts_j
              
                    h5.faceNodes[(k_1*4)+0] = p_offset + i + (j)*self.npts_i + (k)*self.npts_i*self.npts_j
                    h5.faceNodes[(k_1*4)+1] = p_offset + (i) + (j+1)*self.npts_i + (k)*self.npts_i*self.npts_j
                    h5.faceNodes[(k_1*4)+2] = p_offset + (i+1) + (j+1)*self.npts_i + (k)*self.npts_i*self.npts_j
                    h5.faceNodes[(k_1*4)+3] = p_offset + (i+1) + (j)*self.npts_i + (k)*self.npts_i*self.npts_j
              
                    h5.faceNodes[(i_2*4)+0] = p_offset + (i+1) + j*self.npts_i + k*self.npts_i*self.npts_j
                    h5.faceNodes[(i_2*4)+1] = p_offset + (i+1) + (j+1)*self.npts_i + k*self.npts_i*self.npts_j
                    h5.faceNodes[(i_2*4)+2] = p_offset + (i+1) + (j+1)*self.npts_i + (k+1)*self.npts_i*self.npts_j
                    h5.faceNodes[(i_2*4)+3] = p_offset + (i+1) + (j)*self.npts_i + (k+1)*self.npts_i*self.npts_j
              
                    h5.faceNodes[(j_2*4)+0] = p_offset + i + (j+1)*self.npts_i + (k)*self.npts_i*self.npts_j
                    h5.faceNodes[(j_2*4)+1] = p_offset + (i) + (j+1)*self.npts_i + (k+1)*self.npts_i*self.npts_j
                    h5.faceNodes[(j_2*4)+2] = p_offset + (i+1) + (j+1)*self.npts_i + (k+1)*self.npts_i*self.npts_j
                    h5.faceNodes[(j_2*4)+3] = p_offset + (i+1) + (j+1)*self.npts_i + (k)*self.npts_i*self.npts_j
              
                    h5.faceNodes[(k_2*4)+0] = p_offset + i + (j)*self.npts_i + (k+1)*self.npts_i*self.npts_j
                    h5.faceNodes[(k_2*4)+1] = p_offset + (i+1) + (j)*self.npts_i + (k+1)*self.npts_i*self.npts_j
                    h5.faceNodes[(k_2*4)+2] = p_offset + (i+1) + (j+1)*self.npts_i + (k+1)*self.npts_i*self.npts_j
                    h5.faceNodes[(k_2*4)+3] = p_offset + (i) + (j+1)*self.npts_i + (k+1)*self.npts_i*self.npts_j

                    cellID = cellID + 1

        # Offset cell and face references
        h5.nodeVertex = np.append(h5.nodeVertex,self.X,axis=0)
        self.ghostID = ghostID

        
class h5_mesh:
    def __init__(self,ncell=0,nface=0):
        # H5 Attributes
        self.numFaces = nface
        self.numCells = ncell
        # H5 Datasets
        self.cellFace = np.zeros(ncell*6)
        self.faceBC = np.zeros(nface)
        self.faceCell = np.zeros((nface,2))
        self.faceInfo = np.zeros((nface,2))
        self.faceNodes = np.zeros(nface*4)
        self.faceType = np.ones(nface)*4
        self.nodeVertex = np.zeros((0,3))

    def load_zcfd(self,fname):
        # Load zcfd h5 unstructured mesh
        f = h5py.File(fname,"r")
        g = f.get('mesh')
        
        self.numFaces = int(g.attrs.get('numFaces')[0,0])
        self.numCells = int(g.attrs.get('numCells')[0,0])

        self.cellFace = np.array(g.get('cellFace'))
        self.cellType = np.array(g.get('cellType'))
        self.faceBC = np.array(g.get('faceBC'))
        self.faceCell = np.array(g.get('faceCell'))
        self.faceInfo = np.array(g.get('faceInfo'))
        self.faceNodes = np.array(g.get('faceNodes'))
        self.faceType = np.array(g.get('faceType'))
        self.nodeVertex = np.array(g.get('nodeVertex'))

        # faceIndex dataset
        self.faceIndex = np.zeros(self.numFaces,dtype=int)
        for i in range(self.numFaces-1):
            self.faceIndex[i+1] = self.faceIndex[i] + self.faceType[i]
        return
            

    def writetec(self,fname):
        # Convert zcfd mesh to tecplot visualisation format
        # Write ZCFD mesh to tecplot FEPOLYHEDRON FORMAT
        n_v = np.size(self.nodeVertex[:,0])
        n_c = self.numCells
        n_f = self.numFaces
        n_fnodes = np.size(self.faceNodes)

        fout = open(fname,"w")

        print('Writing Header Information')
        fout.write("VARIABLES= \"X\" \"Y\" \"Z\"\n")
        fout.write("ZONE \n")
        fout.write("NODES = {} \n".format(n_v))
        fout.write("FACES = {} \n".format(n_f))
        fout.write("TOTALNUMFACENODES = {} \n".format(n_fnodes))
        fout.write("NUMCONNECTEDBOUNDARYFACES = 0 \n")
        fout.write("TOTALNUMBOUNDARYCONNECTIONS = 0 \n")
        fout.write("ELEMENTS = {} \n".format(n_c))
        fout.write("DATAPACKING = BLOCK \n")
        fout.write("ZONETYPE = FEPOLYHEDRON \n")

        print('Writing Node Vertex Points')
        fout.write('# i Vertex Locations \n')
        for i in range(n_v):
            fout.write("{} \n".format(self.nodeVertex[i,0]))
        fout.write('# j Vertex Locations \n')
        for i in range(n_v):
            fout.write("{} \n".format(self.nodeVertex[i,1]))
        fout.write('# k Vertex Locations \n')
        for i in range(n_v):
            fout.write("{} \n".format(self.nodeVertex[i,2]))

        print('Writing Face Info')
        fout.write('# Number of points per face \n')
        for i in range(n_f):
            fout.write("{} \n".format(self.faceType[i,0]))


        print('Writing Face Nodes')
        fout.write('# Nodes for each face \n')
        for i in range(n_f):
            n_points = int(self.faceType[i,0])
            for j in range(n_points):
                index = i * n_points + j
                fout.write("{} ".format(self.faceNodes[index,0]+1))
            fout.write("\n")

        print('Writing Face Cell Interfaces')
        fout.write('# Left Cells \n')
        for i in range(n_f):
            fout.write("{} \n".format(self.faceCell[i,0]+1))
        fout.write('# Right Cells \n')
        for i in range(n_f):
            if self.faceCell[i,1] < n_c:
                fout.write("{} \n".format(self.faceCell[i,1]+1))
            elif self.faceCell[i,1] >= n_c:
                fout.write("0 \n")
        

    def writeh5(self,fname):
        # Write unstructured data to h5 file
        f = h5py.File(fname,"w")
        h5mesh = f.create_group("mesh")

        self.faceNodes = np.reshape(self.faceNodes,self.numFaces*4)

        h5mesh.attrs.create("numFaces",self.numFaces, shape=(1,1))
        h5mesh.attrs.create("numCells",self.numCells, shape=(1,1))

        h5mesh.create_dataset("cellFace", data=self.cellFace)
        h5mesh.create_dataset("faceBC", data=self.faceBC, shape=(self.numFaces,1))
        h5mesh.create_dataset("faceCell", data=self.faceCell)
        h5mesh.create_dataset("faceInfo", data=self.faceInfo)
        h5mesh.create_dataset("faceNodes", data=self.faceNodes, shape=(self.numFaces*4,1))
        h5mesh.create_dataset("faceType", data=self.faceType, shape=(self.numFaces,1))
        h5mesh.create_dataset("nodeVertex", data=self.nodeVertex)
        return

    def extractSurfaceFaces(self,facetag):
        # Extract unstructured surface faces
        surFace = np.array(np.where(self.faceInfo[:,0]==facetag))[0,:]
        
        n_s = 0
        for f in surFace:
            n_s = n_s + int(self.faceType[f])
        n_v = np.size(self.nodeVertex[:,0])

        X_s = np.zeros((n_s,3))

        ii = 0

        for i in surFace:
            n_points = int(self.faceType[i])
            for j in range(n_points):
                index = self.faceIndex[i] + j

                X_s[ii,0] = self.nodeVertex[self.faceNodes[index],0]
                X_s[ii,1] = self.nodeVertex[self.faceNodes[index],1]
                X_s[ii,2] = self.nodeVertex[self.faceNodes[index],2]
            
                ii = ii + 1

        # Assign datasets
        self.n_s = n_s
        self.n_v = n_v

        self.X_s = X_s
        self.X_v = self.nodeVertex
        return
    

# # Load meshes
# mesh1 = cba_mesh('../../FLOWSOLVER2018/Caradonna_Tung/CT8_125K.blk',V=True)
# mesh2 = cba_mesh('../../FLOWSOLVER2018/Caradonna_Tung/CT8_125K.blk',V=True)

# # Bump mesh2 5D downstream
# for i in mesh2.block:
#     for j in range(mesh2.block[i].npts):
#         mesh2.block[i].X[j,2] = mesh2.block[i].X[j,2] - (mesh1.block[14].X[-1,2] - mesh1.block[0].X[0,2])

# # Change boundary conditions for bottom faces
# for i in range(0,4):
#     mesh1.block[i].connectivity[5,0] = 2
#     mesh1.block[i].connectivity[5,1] = 36 + 14 + i
#     mesh1.block[i].connectivity[5,2] = 5

# for i in range(18,22):
#     mesh1.block[i].connectivity[5,0] = 2
#     mesh1.block[i].connectivity[5,1] = 36 + 32 + i
#     mesh1.block[i].connectivity[5,2] = 5

# for i in range(14,18):
#     mesh2.block[i].connectivity[5,0] = 2
#     mesh2.block[i].connectivity[5,1] = i - 14
#     mesh2.block[i].connectivity[5,2] = 5

# for i in range(32,36):
#     mesh2.block[i].connectivity[5,0] = 2
#     mesh2.block[i].connectivity[5,1] = i - 14
#     mesh2.block[i].connectivity[5,2] = 5

# mesh1.n_blocks = mesh1.n_blocks*2

# for i in mesh2.block:
#     mesh1.block[i+36] = mesh2.block[i]

# mesh1.writeblk('combined.blk')

mesh = h5_mesh()
mesh.load_zcfd('../../data/3D/MDO/MDO_125K.h5')
mesh.extractSurfaceFaces(4)
# mesh.writetec('combined.plt')

