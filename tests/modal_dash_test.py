from zcfdutils.Wrappers import modeReader
from zcfdutils.genericmodalmodel import genericmodalmodel
import numpy as np

mm = modeReader.atomReader(
    "/home/tom.wainwright/cases/IEA_15MW/IEA_15MW", pitch=-22.690
)

# create dummy deformation vector- smooth 90 degree rotation about z with 0 translation
num_nodes = mm.num_grid_points
deformations = np.zeros((num_nodes, 6))
deformations[:, 5] = np.linspace(0, -np.pi / 2, num_nodes)

mm.write_grid_deformed_vtk(deformations)
mm.write_grid_vtk()
mm.write_mode_vtk(scale_factor=10.0)
