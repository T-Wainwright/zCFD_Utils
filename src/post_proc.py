import h5py
import numpy as np

class results():
    def __init__(self,fname='NONE',V=True):
        if fname != 'NONE':
            self.loadH5(fname)

    def loadH5(self,fname,V=True):
            if V:
                print('Loading Results file: {}'.format(fname))
            f = h5py.File(fname,"r")
            if V:
                print('File loaded successfully')

            groups = list(f.keys())
            
            if V:
                print('File has {} groups: {}'.format(len(groups),groups))
            
            if V:
                print(list(f.values()))
            for g in groups:
                d = f.get(g)
                subgroup = list(d.keys())

                if V:
                    print('Group \'{}\' has {} Subgroups: {}'.format(g,len(subgroup),subgroup))

                if V:
                    print(list(d.items()))

                self.solution = np.array(d.get('solution'))
                self.timestepdata = np.array(d.get('timestepdata'))
                self.globalIndex = np.array(d.get('globalIndex'))
                self.globalToLocalIndex = np.array(d.get('globalToLocalIndex'))

class mesh():
    def __init__(self,fname='NONE',V=True):
        if fname != 'NONE':
            self.load_zcfd(fname)
    
    def load_zcfd(self,fname,V=True):
        # Load zcfd h5 unstructured mesh
        if V:
            print('Loading zCFD mesh: {}'.format(fname))
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

# results=results('../../cases/IEA_15_P1_OUTPUT/IEA_15_results.h5',V=True)
# mesh=mesh('../../data/3D/IEA_15MW/IEA_15MW_500K.h5')
# print(np.max(mesh.faceCell))