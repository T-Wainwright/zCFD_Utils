# Some Demo Cases for how to use the modules in mesh_utils

import mesh_utils

# Convert CBA mesh to zcfd mesh
mesh = mesh_utils.CBA_mesh('../data/Omesh.blk')
mesh.convert_h5_data()
mesh.write_h5('../data/Omesh.blk.h5')

# Convert CBA mesh to plot3D (for importing to pointwise)
mesh = mesh_utils.CBA_mesh('../data/Omesh.blk')
mesh.write_p3d('../data/Omesh.blk.p3d')

# Convert zCFD mesh to tecplot
mesh = mesh_utils.zCFD_mesh('../data/Omesh.blk.h5')
mesh.writetec('../data/Omesh.blk.h5.plt')

print('finished')
