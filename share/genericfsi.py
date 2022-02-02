"""
Copyright (c) 2012-2020, Zenotech Ltd
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Zenotech Ltd nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL ZENOTECH LTD BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import numpy as np
from zcfd import MPI
from zcfd.solvers.utils.RuntimeLoader import create_generic_fsi
from zcfd.utils import config
from zcfdutils import py_rbf
from zcfdutils.Wrappers import ATOM_Wrapper


def get_pressure_force(self):
    pressures = np.array(self.fsi.get_pressures(self.solver_data[0], self.mesh[0]))
    normals = self.fsi.get_faceNormals(self.mesh[0])
    normals = np.reshape(normals, (len(pressures), 3))

    gauge_pressure = pressures - np.ones_like(pressures) * self.parameters['ic_2']['pressure']

    pressure_force = np.zeros((len(pressures), 3))

    for i, p in enumerate(gauge_pressure):
        pressure_force[i, :] = normals[i, :] * p

    return pressure_force


def post_init(self):
    self.fsi = create_generic_fsi(self.solverlib, self.parameters['case name'], self.parameters['problem name'], self.parameters)

    # Get rank
    self.rank = MPI.COMM_WORLD.Get_rank()

    # Available getters
    num_faces = self.fsi.init(self.mesh[0])

    # Parallel tasks
    p = self.fsi.get_pressures(self.solver_data[0], self.mesh[0])
    nodes = self.fsi.get_fsi_nodes(self.mesh[0])  # x, y, z of nodes on fsi surface
    face_nodes = self.fsi.get_fsi_face_nodes()  # face nodes ALL QUADS CURRENTLY
    self.num_nodes = int(len(nodes) / 3)
    normals = self.fsi.get_faceNormals(self.mesh[0])
    centres = self.fsi.get_faceCentres(self.mesh[0])

    flat_centres = np.reshape(centres, (int(len(centres) / 3), 3))
    flat_nodes = np.reshape(nodes, ((self.num_nodes, 3)))

    self.atom = ATOM_Wrapper.atom_struct(self.parameters['fsi']['user variables']['blade fe'])
    self.atom.load_modes(self.parameters['fsi']['user variables']['modal model'])

    self.rbf = py_rbf.UoB_coupling(self.atom.struct_nodes, flat_centres)
    self.rbf.generate_transfer_matrix(self.parameters['fsi']['alpha'])

    self.rbf2 = py_rbf.UoB_coupling(flat_nodes, self.atom.struct_nodes)
    self.rbf2.generate_transfer_matrix(self.parameters['fsi']['alpha'])

    if self.rank == 0:
        # rank 0 tasks
        self.node_labels = self.fsi.get_fsi_node_labels()

    MPI.COMM_WORLD.Barrier()

    # RBF pre-processing
    self.fsi.init_morphing(self.mesh[0])


def start_real_time_cycle(self):
    # initialise displacement list
    u = [0.0 * ii for ii in range(self.num_nodes * 3)]
    pressure_force = get_pressure_force(self)
    dt = self.real_time_step
    if self.rank == 0:
        # rank 0 tasks
        structural_force = self.rbf.interp_12(pressure_force)
        disp = self.atom.deform_struct(structural_force)
        aero_disp = self.rbf2.interp_12(disp)
        u = list(aero_disp.flatten())
        

    u = MPI.COMM_WORLD.bcast(u, root=0)
    dt = MPI.COMM_WORLD.bcast(dt, root=0)
    # Perform RBF and mesh updates
    self.fsi.deform_mesh(self.mesh[0], u, dt)


def post_advance(self):
    # initialise displacement list
    u = [0.0 * ii for ii in range(self.num_nodes * 3)]
    pressure_force = get_pressure_force(self)
    dt = self.real_time_step
    if self.rank == 0:
        # rank 0 tasks
        structural_force = self.rbf.interp_12(pressure_force)
        disp = self.atom.deform_struct(structural_force)
        aero_disp = self.rbf2.interp_12(disp)
        u = list(aero_disp.flatten())
        

    u = MPI.COMM_WORLD.bcast(u, root=0)
    dt = MPI.COMM_WORLD.bcast(dt, root=0)
    # Perform RBF and mesh updates
    self.fsi.deform_mesh(self.mesh[0], u, dt)


def post_solve(self):
    if self.rank == 0:
        # rank 0 tasks
        pass


generic_fsi_coupling = {'post init hook': post_init,
                       'start real time cycle hook': start_real_time_cycle,
                       'post advance hook': post_advance,
                       'post solve hook': post_solve}
