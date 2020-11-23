import numpy as np 
import h5py
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
        self.nface = 0                  # Number of faces
        self.ncell = 0                  # Number of cells
        self.f_com = 0                  # Number of common faces

        self.data_path = ''  # Path to data

        self.V = V                      # Verbosity logical flag

        self.block = {}                 # Dictionary of blocks
        self.Common_faces = {}          # Dictionary of common faces
       
        
        
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
            # Case 1- common faces are from same block- 'Wrap around' - Omesh
            if self.Common_faces[i]['block1'] == self.Common_faces[i]['block2']:
                print('wrap')
                # Pair up faces to be wrapped in ascending order
                if self.Common_faces[i]['face1'] > self.Common_faces[i]['face2']:
                    del self.Common_faces[i] 
            # Case 2- common faces are from different blocks- stitch - Standard MB mesh
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

    def stack_rotor(self):
        print('Stacking Rotor')


class cba_block:
    def __init__(self):
        # Integers
        self.blockID = 0                # Block Number
        self.npts = 0                   # Number of points in block
        self.npts_i = 0                 # Number of points in i direction
        self.npts_j = 0                 # Number of points in j direction
        self.npts_k = 0                 # Number of points in k direction

        self.nface = 0                  # Number of faces
        self.nface_i = 0                # Number of i plane faces
        self.nface_j = 0                # Number of j plane faces
        self.nface_k = 0                # Number of k plane faces

        self.ncell = 0                  # Number of cells

        # Arrays
        self.X = np.array((0,3))        # Coordinate points
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

    def load_zcfd(self,fname,V=True):
        # Load zcfd h5 unstructured mesh
        if V:
            print('Loading zCFD mesh: {}'.format(fname))
        f = h5py.File(fname,"r")
        g = f.get('mesh')
        
        self.numFaces = int(g.attrs.get('numFaces'))
        self.numCells = int(g.attrs.get('numCells'))

        self.cellZone = np.array(g.get('cellZone'))
        self.cellFace = np.array(g.get('cellFace'))
        self.cellType = np.array(g.get('cellType'))
        self.faceBC = np.array(g.get('faceBC'))
        self.faceCell = np.array(g.get('faceCell'))
        self.faceInfo = np.array(g.get('faceInfo')[0])
        self.faceNodes = np.array(g.get('faceNodes'))
        self.faceType = np.array(g.get('faceType'))
        self.nodeVertex = np.array(g.get('nodeVertex'))

        # faceIndex dataset
        self.faceIndex = np.zeros(self.numFaces,dtype=int)
        for i in range(self.numFaces-1):
            self.faceIndex[i+1] = self.faceIndex[i] + self.faceType[i]

        if V:
            print('zCFD mesh successfully loaded ')
            print('nCells= {} \t nFaces= {}'.format(self.numCells,self.numFaces))

        return

            

    def writetec(self,fname,V=True):
        # Write ZCFD mesh to tecplot FEPOLYHEDRON FORMAT
        if V:
            print('Writing tecplot mesh file: {}'.format(fname))
        n_v = np.size(self.nodeVertex[:,0])
        n_c = self.numCells
        n_f = self.numFaces
        n_fnodes = np.size(self.faceNodes)

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
            fout.write("{} \n".format(self.nodeVertex[i,0]))
        fout.write('# j Vertex Locations \n')
        for i in range(n_v):
            fout.write("{} \n".format(self.nodeVertex[i,1]))
        fout.write('# k Vertex Locations \n')
        for i in range(n_v):
            fout.write("{} \n".format(self.nodeVertex[i,2]))

        if V:
            print('Writing Face Info')
        fout.write('# Number of points per face \n')
        for i in range(n_f):
            fout.write("{} \n".format(self.faceType[i][0]))
<<<<<<< HEAD
=======

        if V:
            print('Writing Face Nodes')
        fout.write('# Nodes making up each face \n')
        for i in range(n_f):
            n_points = int(self.faceType[i])
            for j in range(n_points):
                index = i * n_points + j
                fout.write("{} ".format(self.faceNodes[index,0]+1))
            fout.write("\n")

        if V:
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
        
        if V:
            print('tecplot file written successfully')

    def writetec_boundary(self,fname,V=True):
        # Write ZCFD mesh to tecplot FEPOLYHEDRON FORMAT
        if V:
            print('Writing tecplot mesh file: {}'.format(fname))
        n_v = np.size(self.nodeVertex[:,0])
        n_c = self.numCells
        n_f = self.numFaces
        n_fnodes = np.size(self.faceNodes)

        fout = open(fname,"w")
        if V:
            print('Writing Header Information')
        fout.write("VARIABLES= \"X\" \"Y\" \"Z\" \"t\" \n")
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
            fout.write("{} \n".format(self.nodeVertex[i,0]))
        fout.write('# j Vertex Locations \n')
        for i in range(n_v):
            fout.write("{} \n".format(self.nodeVertex[i,1]))
        fout.write('# k Vertex Locations \n')
        for i in range(n_v):
            fout.write("{} \n".format(self.nodeVertex[i,2]))

        if V:
            print('Writing Face Info')
        fout.write('# Number of points per face \n')
        for i in range(n_f):
            fout.write("{} \n".format(int(self.faceType[i][0])))
>>>>>>> ad54f0d988afbd28d1fc6c8242256ea7e62c2f28

        if V:
            print('Writing Face Nodes')
        fout.write('# Nodes making up each face \n')
        for i in range(n_f):
            n_points = int(self.faceType[i])
            for j in range(n_points):
                index = i * n_points + j
                fout.write("{} ".format(self.faceNodes[index,0]+1))
            fout.write("\n")

        if V:
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
        
        if V:
            print('tecplot file written successfully')
        

    def writeh5(self,fname,V=True):
        # Write unstructured data to h5 file
        if V:
            print('Writing h5 mesh file: {}'.format(fname))
        f = h5py.File(fname,"w")
        h5mesh = f.create_group("mesh")

        self.faceNodes = np.reshape(self.faceNodes,self.numFaces*4)

        h5mesh.attrs.create("numFaces",self.numFaces, shape=(1,1))
        h5mesh.attrs.create("numCells",self.numCells, shape=(1,1))

        # h5mesh.create_dataset("cellFace", data=self.cellFace)
        h5mesh.create_dataset("faceBC", data=self.faceBC, shape=(self.numFaces,1))
        h5mesh.create_dataset("faceCell", data=self.faceCell)
        h5mesh.create_dataset("faceInfo", data=self.faceInfo)
        h5mesh.create_dataset("faceNodes", data=self.faceNodes, shape=(self.numFaces*4,1))
        h5mesh.create_dataset("faceType", data=self.faceType, shape=(self.numFaces,1))
        h5mesh.create_dataset("nodeVertex", data=self.nodeVertex)

        if V:
            print('h5 file written successfully')
        return

    def write_surface_tec(self, V=True):
        # Extract unstructured surface faces
        if V:
            print('Extracting faces with tag: {}'.format(facetag))
        
        # Get surface face ID's
        surface_faceID = np.array(np.where(self.faceInfo[:,0]!=0))[0,:]
        n_face = len(surface_faceID)
        surface_faceTag = np.zeros(n_face)
        surface_faceNodes = np.zeros(n_face*4)

        # Find nodes and boundary tags
        for i in range(n_face):
            surface_faceTag[i] = self.faceInfo[[surface_faceID[i]],0]
            for j in range(self.faceType[surface_faceID[i]]):
                index = 4*i + j
                surface_faceNodes[index] = self.faceNodes[4*surface_faceID[i]+j,0]
        
        # Extract only unique nodes
        unique_nodes, unique_counts = np.unique(surface_faceNodes,return_counts=True)
        n_nodes=len(unique_nodes)


        f = open("../data/Test_surface.plt","w")
        f.write("TITLE = Boundary plot\n")
        f.write("VARIABLES = \"X\" \"Y\" \"Z\" \"Tag\"\n")
        f.write("ZONE T=\"PURE-QUADS\", NODES={}, ELEMENTS={}, DATAPACKING=BLOCK, VARLOCATION=([4]=CELLCENTERED), ZONETYPE=FEQUADRILATERAL\n".format(n_nodes,n_face))

        # Print unique node locations
        for n in unique_nodes:
            f.write("{}\n".format(self.nodeVertex[int(n),0]))
        for n in unique_nodes:
            f.write("{}\n".format(self.nodeVertex[int(n),1]))
        for n in unique_nodes:
            f.write("{}\n".format(self.nodeVertex[int(n),2]))
        for face in range(n_face):
            f.write("{}\n".format(int(surface_faceTag[face])))

        # Print nodes making up each face
        for face in range(n_face):
            for i in range(4):
                f.write("{} ".format(np.where(unique_nodes==surface_faceNodes[face*4+i])[0][0]+1))
            f.write("\n")

        return

    def extractHaloFaces(self):

        # Count number of halo cells in the mesh
        max_cell_ID = max(self.faceCell[:,1])
        num_halo_cells = max_cell_ID - self.numCells
        print('num_Halo_cells= {}'.format(num_halo_cells))

        # Count the number of faces requiring a halo cell
        num_internal_faces = np.count_nonzero(self.faceBC==0)
        num_halo_faces = self.numFaces - num_internal_faces
        print('num_halo_faces= {}'.format(num_halo_faces))

    def writeLKEconf(self,fname,r0,nbase,V=True):
        # Output .conf files for LKE mesh deformation program
        if V:
            print('Writing LKE mesh deformation config files')
        volfile = 'volume'
        surfile = 'surface'
        dformat = 'xyz'

        fout = open(fname,"w")
        fout.write("voltype = {} \n".format(dformat))
        fout.write("mesh = {} \n".format(volfile + '.' + dformat))
        fout.write("surfpts = {} \n".format(surfile + '.' + dformat))
        fout.write("rbfmode = 1 \n")
        fout.write("r0 = {} \n".format(r0))
        fout.write("nbase = {}".format(nbase))

        fout.close()

        fvol = open(volfile + '.' + dformat,"w")
        fvol.write("{} \n".format(self.n_v))
        for i in range(self.n_v):
            fvol.write("{:.12f} \t {:.12f} \t {:.12f} \n".format(self.X_v[i,0],self.X_v[i,1],self.X_v[i,2]))
        fvol.close()

        fsur = open(surfile + '.' + dformat,"w")
        fsur.write("{} \n".format(self.n_s))
        for i in range(self.n_s):
            fsur.write("{:.12f} \t {:.12f} \t {:.12f} \n".format(self.X_s[i,0],self.X_s[i,1],self.X_s[i,2]))
        fsur.close()

        if V:
            print('LKE config files written successfully')
        return

    def generate_deformation_file(self,fname):

        fout = open(fname,"w")
        fout.write("{} \n".format(self.n_s))
        for i in range(self.n_s):
            fout.write("{:.12f} \t {:.12f} \t {:.12f} \n".format(self.X_def[i,0],self.X_def[i,1],self.X_def[i,2]))
        fout.close()

    def rotate_surface(self,alpha):
        self.X_def = np.zeros_like(self.X_s)

        alpha_rad = np.deg2rad(alpha)

        R = np.array([[np.cos(alpha_rad), 0,-np.sin(alpha_rad)],[0,1,0],[np.sin(alpha_rad),0,np.cos(alpha_rad)]])

        for i in range(self.n_s):
            self.X_def[i,:] = np.matmul(R,self.X_s[i,:]) - self.X_s[i,:]
            # self.X_def[i,0] = np.cos(alpha_rad) * self.X_s[i,0] - np.sin(alpha_rad) * self.X_s[i,2]
            # self.X_def[i,2] = np.sin(alpha_rad) * self.X_s[i,0] + np.cos(alpha_rad) * self.X_s[i,2]
        # self.X_def = self.X_s

    def translate_surface(self,vec):

        for i in range(self.n_s):
            self.X_def[i,0] = vec[0]
            self.X_def[i,1] = vec[1]
            self.X_def[i,2] = vec[2]

    
    def rbf_rotate(self,surfID,r0,nbase,alpha):
        # Preprocessing
        self.extractSurfaceFaces(surfID)
        self.writeLKEconf('rotate.LKE',r0,nbase)

        os.system('meshprep rotate.LKE')

        self.rotate_surface(alpha)
        # vec = [0,0,5]
        # self.translate_surface(vec)

        self.generate_deformation_file('surface_deformations.xyz')

        os.system('meshdef volume.xyz.meshdef surface_deformations.xyz volume_deformations.xyz')

        self.X_vol_deformed = np.loadtxt('volume_deformations.xyz',skiprows=1)

        print(self.faceType.shape,self.numFaces)
        self.nodeVertex = self.X_vol_deformed

        self.writetec('test.plt')

        self.writeh5('deformed.h5')

        os.system('rm rotate.LKE surface.xyz volume.xyz surface_deformations.xyz volume_deformations.xyz volume.xyz.meshdef def.xyz')

        
# mesh = cba_mesh(fname='../../CBA_meshes/IEA_15MW/IEA_15MW_5M.blk',V=True)

# f = open('../../CBA_meshes/IEA_15MW/IEA_15MW_5M.p3d',"w")

# f.write('{}\n'.format(mesh.n_blocks))
# for b in range(mesh.n_blocks):
#     f.write('{} \t {} \t {}\n'.format(mesh.block[b].npts_i,mesh.block[b].npts_j,mesh.block[b].npts_k))

# for b in range(mesh.n_blocks):
#     for i in range(mesh.block[b].npts):
#         f.write('{}\n'.format(mesh.block[b].X[i,0]))
#     for i in range(mesh.block[b].npts):
#         f.write('{}\n'.format(mesh.block[b].X[i,1]))
#     for i in range(mesh.block[b].npts):
#         f.write('{}\n'.format(mesh.block[b].X[i,2]))
    

# f.close()

mesh = h5_mesh()
<<<<<<< HEAD
mesh.load_zcfd('../../CBA_meshes/IEA_15MW/IEA_15MW_Occluded_12M.cas.h5')
mesh.writetec('../../CBA_meshes/IEA_15MW/IEA_15MW_Occluded_12M.cas.h5.plt')





# mesh1 = cba_mesh('../../data/3D/IEA_15MW/IEA_15MW_1M.blk')
# mesh2 = cba_mesh('../../data/3D/IEA_15MW/IEA_15MW_1M.blk')

# print(mesh1.block[0].X[0][2])
# print(mesh1.block[14].X[-1][2])

# h = np.abs(mesh1.block[0].X[0][2] - mesh1.block[14].X[-1][2])

# print(h)

# for b in mesh2.block:
#     print(b)
#     # Drop z coordinate by height of mesh1
#     for p in range(mesh2.block[b].npts):
#         mesh2.block[b].X[p][2] = mesh2.block[b].X[p][2] - h

#     # Increase block indexing in connectivity
#     for c in range(6):
#         if mesh2.block[b].connectivity[c,1] != 0:
#             mesh2.block[b].connectivity[c,1] = mesh2.block[b].connectivity[c,1] + 36
    
# # Adjust connectivity
# # Bottom blocks of mesh1:
# print(mesh1.block[1].connectivity)

# connectivity_dict = {0:14,1:15,2:16,3:17,18:32,19:33,20:34,21:35}


# for a in connectivity_dict:
#     mesh1.block[a].connectivity[2,0] = 2                                    # Internal face
#     mesh1.block[a].connectivity[2,1] = connectivity_dict[a] + 37            # Neighbour
#     mesh1.block[a].connectivity[2,2] = 4                                    # Connected to jmax

#     mesh2.block[connectivity_dict[a]].connectivity[3,0] = 2
#     mesh2.block[connectivity_dict[a]].connectivity[3,1] = a + 1
#     mesh2.block[connectivity_dict[a]].connectivity[3,2] = 3

# # Append mesh1 file
# for b in mesh2.block:
#     mesh1.block[b + 36] = mesh2.block[b]

# mesh1.n_blocks = mesh1.n_blocks * 2

# mesh1.writeblk('../../data/3D/IEA_15MW/IEA_15MW_1M_Occluded.blk')
# mesh1.writetec('../../data/3D/IEA_15MW/IEA_15MW_1M_Occluded.blk.plt')
# print(mesh2.block[14].connectivity)
# print(mesh2.block[15].connectivity)
# print(mesh2.block[16].connectivity)
# print(mesh2.block[17].connectivity)
=======
mesh.load_zcfd('../../data/3D/IEA_15MW/Single_Turbine_5M/IEA_15MW_5M.cas.h5')
# mesh.load_zcfd('../data/CT0_250K.blk.h5')
# mesh.writetec(fname='../../data/3D/IEA_15MW/Single_Turbine_5M/IEA_15MW_5M.cas.h5.plt')
>>>>>>> ad54f0d988afbd28d1fc6c8242256ea7e62c2f28

print(mesh.cellZone[0])

f = open('../data/zones.txt',"w")
f.write('{}'.format(np.unique(mesh.cellZone)))
f.close()
print(np.unique(mesh.cellZone))
mesh.writeh5('../data/IEA_15MW_5M.cas.h5')


