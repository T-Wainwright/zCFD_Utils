import scipy.io
import numpy as np

data = scipy.io.loadmat('../data/ModalModel.mat')
blade_fe = scipy.io.loadmat('../data/Blade_FE.mat')

blade_fe = blade_fe['Blade_FE']
ModalStruct = data['ModalStruct']

Struct_nodes = blade_fe['BeamAxis_123'][0][0]
nNodes = Struct_nodes.shape[0]

nEigval = ModalStruct['nEigval'][0][0][0][0]
Eigvec = ModalStruct['Eigvec'][0][0]
KmodalInv = ModalStruct['KmodalInv'][0][0]

Fnodal = np.zeros([Eigvec.shape[0] + 6, 1])
Fnodal[-17] = 1000

# Convert Nodal forces to Modal forces

FModal = np.matmul(np.transpose(Eigvec[:, 0:nEigval]), Fnodal[6:])

QDisp = np.matmul(KmodalInv, FModal)
Disp = np.zeros([Eigvec.shape[0]])

for imode in range(nEigval):
    Disp = Disp + QDisp[imode] * Eigvec[:, imode]

Disp = np.concatenate((np.zeros(6), Disp))
Disp = np.reshape(Disp, (6, nNodes))
