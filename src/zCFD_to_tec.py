import numpy as np 
import h5py

""" 
Code to convert zCFD .h5 mesh format files into tec360 .plt format

Output file will be a .plt file in FEPOLYHEDRON format.

Tom Wainwright 2020
University of Bristol
tom.wainwright@bristol.ac.uk

"""

fname = '../data/Omesh..blk.h5'
tecname = '../data/Omesh.blk.h5.plt'


f = h5py.File(fname,"r")
g = f.get('mesh')

# Read ALL data_sets
cellFace = np.array(g.get('cellFace'))
cellType = np.array(g.get('cellType'))
faceBC = np.array(g.get('faceBC'))
faceCell = np.array(g.get('faceCell'))
faceInfo = np.array(g.get('faceInfo'))
faceNodes = np.array(g.get('faceNodes'))
faceType = np.array(g.get('faceType'))
nodeVertex = np.array(g.get('nodeVertex'))

l = list(g.attrs.keys())

n_v = np.size(nodeVertex[:,0])
n_c = g.attrs.get('numCells')[0,0]
n_f = g.attrs.get('numFaces')[0,0]

n_fnodes = np.size(faceNodes)

print('h5 Mesh Read successfully! \n')
print('numCells = {} \t numFaces = {}'.format(n_c,n_f))
print('Writing .plt file \n')

fout = open(tecname,"w")

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
    fout.write("{} \n".format(nodeVertex[i,0]))
fout.write('# j Vertex Locations \n')
for i in range(n_v):
    fout.write("{} \n".format(nodeVertex[i,1]))
fout.write('# k Vertex Locations \n')
for i in range(n_v):
    fout.write("{} \n".format(nodeVertex[i,2]))


print('Writing Face Info')
fout.write('# Number of points per face \n')
for i in range(n_f):
    fout.write("{} \n".format(faceType[i,0]))


print('Writing Face Nodes')
fout.write('# Nodes for each face \n')
for i in range(n_f):
    n_points = int(faceType[i,0])
    for j in range(n_points):
        index = i * n_points + j
        fout.write("{} ".format(faceNodes[index,0]+1))
    fout.write("\n")

print('Writing Face Cell Interfaces')
fout.write('# Left Cells \n')
for i in range(n_f):
    fout.write("{} \n".format(faceCell[i,0]+1))
fout.write('# Right Cells \n')
for i in range(n_f):
    if faceCell[i,1] < n_c:
        fout.write("{} \n".format(faceCell[i,1]+1))
    elif faceCell[i,1] >= n_c:
        fout.write("0 \n")


