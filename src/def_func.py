import numpy as np 
import matplotlib.pyplot as plt 
import h5py
import os
from mpl_toolkits.mplot3d import Axes3D

class mesh():
    def __init__(self,fname):
        # Load  
        self.X_s = []
        self.X_v = []
        self.n_s = 0
        self.n_v = 0
        self.aux = {}

        self.data_path = '../../data/3D/CT_rotor/'

        self.load_mesh(self.data_path + fname)
    
    # Data loading stuff

    def load_mesh(self,fname):
        print('Loading Mesh')
        ext = fname.split('.')[-1]
        if ext == 'h5':
            print('Identified zCFD h5 Mesh')
            self.load_zcfd(fname)
        elif ext == 'blk':
            print('Identified CBA blk Mesh')
            self.load_cba(fname)
        print('Mesh Loaded')
        return 
        
    
    def load_zcfd(self,fname):
        self.aux['ext'] = 'h5'
        self.aux['fname'] = fname
        f = h5py.File(fname,'r')
        g = f.get('mesh')
        
        # print(list(g.keys()))

        nodeVertex = np.array(g.get('nodeVertex'))
        faceInfo = np.array(g.get('faceInfo'))

        
        surfFace = np.array(np.where(faceInfo[:,0]==4))[0,:]
        print(surfFace)
        faceType = np.array(g.get('faceType'))
        faceNodes = np.array(g.get('faceNodes'))

        # Find surface points
        self.n_s = np.size(surfFace) * 4
        self.n_v = np.size(nodeVertex[:,0])
        print(self.n_s)

        self.X_s = np.zeros([self.n_s,3])
        ii = 0
        for i in surfFace:
            n_points = int(faceType[i])
            # print(n_points)
            for j in range(n_points):
                index = i*n_points + j
                # print(index)
                self.X_s[ii,0] = nodeVertex[faceNodes[index],0]
                self.X_s[ii,1] = nodeVertex[faceNodes[index],1]
                self.X_s[ii,2] = nodeVertex[faceNodes[index],2]
                
                ii = ii+1

        # Find remaining volume points
        self.X_v = nodeVertex
        return

    def load_cba(self,fname):
        self.aux['ext'] = 'blk'
        self.aux['fname'] = fname
        DATA = np.loadtxt(fname,dtype='f')
        npts_i = int(DATA[1,0]) # Number of surface points
        npts_j = int(DATA[1,1]) # Number of radial points
        npts_k = int(DATA[1,2]) # K = 1 for a 2D mesh

        self.aux['npts_i'] = npts_i
        self.aux['npts_j'] = npts_j
        self.aux['npts_k'] = npts_k

        self.n_v = npts_i * npts_j * npts_k
        self.n_s = npts_i * npts_k

        self.X_s = DATA[2:self.n_s+2,:]
        self.X_v = DATA[2:self.n_v+2,:]
        return
    
    # RBF Stuff

    def rbf_coeff(self,r_s):
        phi = self.phi_gen(r_s)
        psi = self.psi_gen(r_s)

        print('Calculating Transfer Matrix')
        self.H = []
        self.H = np.matmul(psi, np.linalg.pinv(phi))

        return 

    def phi_gen(self,r_s):  
        print('Calculating Phi')
        phi = np.zeros([self.n_s,self.n_s])       
        for i in range(self.n_s):
            for j in range(self.n_s):
                r = np.linalg.norm(self.X_s[i,:]-self.X_s[j,:])/r_s

                phi[j,i] = c3(r)
        return phi

    def multiscale_seq(self,r_s):
        print('Sequencing control points')
        n_base = 10
        # Generate required data sets
        active_list = np.zeros(1,dtype=int)
        inactive_list = list(range(1,self.n_s))
        r_max = np.ones_like(inactive_list,dtype=float) * 100000
        r_max[0] = 0
        r_support = r_s
        n_active = np.size(active_list)
        n_inactive = np.size(inactive_list)

        while n_active < self.n_s:
            k_ID = active_list[-1] # ID of last point added to list
            for i in range(n_inactive):
                l_ID = inactive_list[i]
                r = np.linalg.norm(self.X_s[k_ID,:]-self.X_s[l_ID,:])

                # Update largest radii for each point
                if r < r_max[i]:
                    r_max[i] = r

            max_loc = np.argmax(r_max)          # Location of maximum radius in inactive list
            max_ID = inactive_list[max_loc]     # Node ID of point with largest radius

            # Add point with largest separation to active set, update lists
            active_list = np.append(active_list,max_ID)
            
            if n_active <= n_base:
                r_support = np.append(r_support,r_s)
            else:
                r_support = np.append(r_support,r_max[max_loc])

            inactive_list = np.delete(inactive_list,max_loc)    
            r_max = np.delete(r_max,max_loc)
                
            
            # Update number of active vs inactive
            n_active = np.size(active_list)
            n_inactive = np.size(inactive_list)
            print(n_active)
        
        # Save active list
        self.active_list = active_list
        self.r_support = r_support
        return

    def multiscale_coef(self,r_s):
        print('Calculating coefficients')
        # Explicit variables
        n_base = 10
        # Implicit variablaes
        n_reduced = self.n_s - n_base
        phi_b = np.zeros((n_base,n_base),dtype=float)
        psi_r = np.zeros((n_reduced,n_base),dtype=float)
        L = np.zeros((n_reduced,n_reduced),dtype=float)
        
        # phi_b
        for i in range(n_base):
            for j in range(i):
                r = np.linalg.norm(self.X_s[self.active_list[i],:]-self.X_s[self.active_list[j]])
                e = r/self.r_support[i]
                if np.isnan(e):
                    print('NAN1')
                if e >= 1:
                    coef = 0
                else:
                    coef = c2(e)
                phi_b[i,j] = coef
            phi_b[i,i] = 1

        # psi_r
        for i in range(n_reduced):
            for j in range(n_base):
                r = np.linalg.norm(self.X_s[self.active_list[i+n_base],:]-self.X_s[self.active_list[j],:])
                if self.r_support[i+n_base] == 0 and r == 0:
                    coef = 1
                else:
                    e = r/self.r_support[i+n_base]
                    if e >= 1:
                        coef = 0
                    else:
                        coef = c2(e)
                psi_r[i,j] = coef
                
        # L
        for i in range(n_reduced):
            for j in range(i):
                r = np.linalg.norm(self.X_s[self.active_list[i+n_base],:]-self.X_s[self.active_list[j+n_base],:])
                e = r/self.r_support[i+n_base]
                if np.isnan(e):
                    print('NAN3')
                if e >= 1:
                    coef = 0
                else:
                    coef = c2(e)
                L[i,j] = coef     
            L[i,i] = 1
                    

        print(phi_b)



        return

    def psi_gen(self,r_s):
        print('Calculating Psi')
        psi = np.zeros([self.n_v,self.n_s])

        for i in range(self.n_s):
            for n in range(self.n_v):
                r = np.linalg.norm(self.X_s[i,:]-self.X_v[n,:])/r_s

                psi[n,i] = c3(r)
        return psi

    # Visualisation Stuff

    def write2tec(self,filename):
        print('Writing tecplot output')
        filename = self.data_path + filename
        # Examine file extension
        if self.aux['ext'] == 'blk':
            # Write BLK file to tec
            self.blk2tec(filename)
        if self.aux['ext'] == 'h5':
            self.h52tec(filename)
        return
    
    def blk2tec(self,filename):
         # Write BLK file to tec
        fout = open(filename,"w")
        fout.write("VARIABLES= \"X\" \"Y\" \"Z\"\n")
        fout.write("ZONE I=   {} J=   {} ,K={},F=POINT\n".format(self.aux['npts_i'],self.aux['npts_j'],self.aux['npts_k']))
        for i in range(np.size(self.X_v[:,0])):
            fout.write("{}  \t{}  \t{}\n".format(self.X_v[i,0],self.X_v[i,1],self.X_v[i,2]))
        return

    def h52tec(self,filename):
        print('Writing h5 to tecplot')
        # Write CBA file to tec
        f = h5py.File(self.aux['fname'],"r")
        g = f.get('mesh')

        # Read ALL data_sets
        faceCell = np.array(g.get('faceCell'))
        faceNodes = np.array(g.get('faceNodes'))
        faceType = np.array(g.get('faceType'))
        nodeVertex = self.X_v

        n_v = np.size(nodeVertex[:,0])
        n_c = g.attrs.get('numCells')[0,0]
        n_f = g.attrs.get('numFaces')[0,0]

        n_fnodes = np.size(faceNodes)

        print('h5 Mesh Read successfully! \n')
        print('numCells = {} \t numFaces = {}'.format(n_c,n_f))
        print('Writing .plt file \n')

        fout = open(filename,"w")

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
        for i in range(n_v):
            fout.write("{} \n".format(nodeVertex[i,0]))
        for i in range(n_v):
            fout.write("{} \n".format(nodeVertex[i,1]))
        for i in range(n_v):
            fout.write("{} \n".format(nodeVertex[i,2]))


        print('Writing Face Info')
        for i in range(n_f):
            fout.write("{} \n".format(faceType[i,0]))


        print('Writing Face Nodes')
        for i in range(n_f):
            n_points = int(faceType[i,0])
            for j in range(n_points):
                index = i * n_points + j
                fout.write("{} ".format(faceNodes[index,0]+1))
            fout.write("\n")
        for i in range(n_f):
            fout.write("{} \n".format(faceCell[i,0]+1))


        print('Writing Face Cell Interfaces')
        for i in range(n_f):
            if faceCell[i,1] < n_c:
                fout.write("{} \n".format(faceCell[i,1]+1))
            elif faceCell[i,1] >= n_c:
                fout.write("0 \n")
        return  
    
    # Deformation Stuff
    def transfunc(self):
        print('Calculating surface deformations')
        # Mesh transformation function
        u_x = np.zeros(self.n_s)
        u_y = np.zeros(self.n_s)
        u_z = np.zeros(self.n_s)

        t = np.deg2rad(0)
        
        # for i in range(self.n_s):
        #     u_x_tr = 0
        #     u_y_tr = 0
        #     u_z_tr = 5

        #     u_x_rot = (np.cos(t)*self.X_s[i,0] - np.sin(t)*self.X_s[i,2]) - self.X_s[i,0]
        #     u_z_rot = (np.sin(t)*self.X_s[i,0] + np.cos(t)*self.X_s[i,2]) - self.X_s[i,2]

        #     u_x[i] = u_x_tr + u_x_rot
        #     u_y[i] = u_y_tr
        #     u_z[i] = u_z_tr + u_z_rot
        
        self.u_s = np.zeros_like(self.X_s)

        for i in range(self.n_s):
            # u_z[i] = 0.25 * self.X_s[i,1]**2
            u_z[i] = (0.02 / 24) * self.X_s[i,1]**2

        self.u_s[:,0] = u_x
        self.u_s[:,1] = u_y
        self.u_s[:,2] = u_z


        return 

    def rbf_interp(self):
        print('Interpolating surface deformations to volume mesh')
        # Transfer movement of surface mesh to volume mesh
        u_v = np.zeros_like(self.X_v)

        u_v[:,0] = np.matmul(self.H,self.u_s[:,0])
        u_v[:,1] = np.matmul(self.H,self.u_s[:,1])
        u_v[:,2] = np.matmul(self.H,self.u_s[:,2])

        self.X_v[:,0] = self.X_v[:,0] + u_v[:,0]
        self.X_v[:,1] = self.X_v[:,1] + u_v[:,1]
        self.X_v[:,2] = self.X_v[:,2] + u_v[:,2]

        print('RBF Mesh Deformation Complete')
        return

    def writeLKEconf(self,fname,r0,nbase):
        print('Writing LKE config file')
        volfile = 'volume'
        surfile = 'surface'
        dformat = 'xyz'


        fname = self.data_path + fname

        fout = open(fname,"w")
        fout.write("voltype = {} \n".format(dformat))
        fout.write("mesh = {} \n".format(self.data_path + volfile + '.' + dformat))
        fout.write("surfpts = {} \n".format(self.data_path + surfile + '.' + dformat))
        fout.write("rbfmode = 1 \n")
        fout.write("r0 = {} \n".format(r0))
        fout.write("nbase = {}".format(nbase))

        fout.close()

        fvol = open(self.data_path + volfile + '.' + dformat,"w")
        fvol.write("{} \n".format(self.n_v))
        for i in range(self.n_v):
            fvol.write("{:.12f} \t {:.12f} \t {:.12f} \n".format(self.X_v[i,0],self.X_v[i,1],self.X_v[i,2]))
        fvol.close()

        fsur = open(self.data_path + surfile + '.' + dformat,"w")
        fsur.write("{} \n".format(self.n_s))
        for i in range(self.n_s):
            fsur.write("{:.12f} \t {:.12f} \t {:.12f} \n".format(self.X_s[i,0],self.X_s[i,1],self.X_s[i,2]))


        return
    
    def writeLKEdef(self,fname):
        print('Writing LKE deformation file')
        fname = self.data_path + fname
        self.transfunc()
        fdef = open(fname,"w")
        fdef.write("{} \n".format(self.n_s))
        for i in range(self.n_s):
            fdef.write("{:.12f} \t {:.12f} \t {:.12f} \n".format(self.u_s[i,0],self.u_s[i,1],self.u_s[i,2]))
        return
    
    def update(self,fname):
        DATA = np.loadtxt(fname,dtype=float,skiprows=1)
        self.X_v = DATA
        return


def c0(r):
    f = (1-r)**2
    return f

def c2(e):
    f = ((1-e)**4)*(4*e+1)
    return f

def c3(r):
    f = (1-r)**6 * (35*r**2 + 18*r + 3)/3
    return f



mesh = mesh('CT0_500K.h5')
mesh.transfunc()
mesh.writeLKEconf('out.conf',100,200)
os.system('meshprep ../../data/out.conf')
mesh.writeLKEdef('def.xyz')
os.system('meshdef ../../data/volume.xyz.meshdef ../../data/def.xyz output.xyz')
mesh.update('output.xyz')
mesh.write2tec('deformed.plt')
print('finished')


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(mesh.X_s[:,0],mesh.X_s[:,1],mesh.X_s[:,2])
# plt.show()