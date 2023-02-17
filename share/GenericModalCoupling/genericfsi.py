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
from zutil.libmultiscale import multiscale
# from zcfd.utils.GenericModeReader import GenericModeReader
# from zcfd.utils.NastranReader import NastranReader
from zcfd.utils.coupling.genericmodalmodel import genericmodalmodel
from zcfdutils.Wrappers.modeReader import cba_modal


def get_pressure_force(self):
    pressures = np.array(self.fsi.get_pressures(
        self.solver_data[0], self.mesh[0]))
    normals = self.fsi.get_faceNormals(self.mesh[0])
    normals = np.reshape(normals, (len(pressures), 3))

    gauge_pressure = pressures - \
        np.ones_like(pressures) * self.parameters['ic_1']['pressure']

    pressure_force = np.zeros((len(pressures), 3))

    for i, p in enumerate(gauge_pressure):
        pressure_force[i, :] = normals[i, :] * p

    return pressure_force


def post_init(self):
    print('Post init')
    self.fsi = create_generic_fsi(
        self.solverlib, self.parameters['case name'], self.parameters['problem name'], self.parameters)

    # Get rank
    self.rank = MPI.COMM_WORLD.Get_rank()

    # Available getters
    num_faces = self.fsi.init(self.mesh[0])

    # Parallel tasks
    p = self.fsi.get_pressures(self.solver_data[0], self.mesh[0])
    # x, y, z of nodes on fsi surface
    nodes = self.fsi.get_fsi_nodes(self.mesh[0])
    face_nodes = self.fsi.get_fsi_face_nodes()  # face nodes ALL QUADS CURRENTLY
    
    self.num_nodes = int(len(nodes) / 3)
    normals = self.fsi.get_faceNormals(self.mesh[0])
    centres = self.fsi.get_faceCentres(self.mesh[0])

    flat_centres = np.reshape(centres, (int(len(centres) / 3), 3))
    flat_nodes = np.reshape(nodes, ((self.num_nodes, 3)))

    # nastran_modal_model = NastranReader(self.parameters['fsi']['user variables']['nastran file name'])
    # nastran_modal_model.read_grid_points()
    # nastran_modal_model.read_modes(self.parameters[])
    CBA_modal_model = cba_modal(self.parameters["fsi"]["user variables"]["cba filename"])
    CBA_modal_model.calculate_mode_frequencies()
    # self.genericmodalmodel = genericmodalmodel(nastran_modal_model.grid_points, nastran_modal_model.mode_freqs, nastran_modal_model.mode_data)
    self.genericmodalmodel = genericmodalmodel(CBA_modal_model.grid, CBA_modal_model.mode_frequencies, CBA_modal_model.eigenvectors, mode_list=self.parameters['fsi']['user variables']['mode list'])

    # Rank 0 tasks
    if self.rank == 0:
        self.multiscale = multiscale(flat_centres, self.parameters['fsi']['base point fraction'], self.parameters['fsi']['alpha'])
        self.multiscale.sample_control_points(False)

        self.multiscale2 = multiscale(self.genericmodalmodel.grid_points, self.parameters['fsi']['base point fraction'], self.parameters['fsi']['alpha'])
        self.multiscale2.sample_control_points(False)

        self.multiscale.preprocess_V(self.genericmodalmodel.grid_points)
        self.multiscale2.preprocess_V(flat_nodes)

    MPI.COMM_WORLD.Barrier()

    # RBF pre-processing
    self.fsi.init_morphing(self.mesh[0])


def start_real_time_cycle(self):
    # # initialise displacement list
    # u = [0.0 * ii for ii in range(self.num_nodes * 3)]
    # # self.u_total = u
    # pressure_force = get_pressure_force(self)
    # dt = self.real_time_step
    # dt = 1
    # if self.rank == 0:
    #     # rank 0 tasks
    #     self.multiscale.multiscale_solve(pressure_force)
    #     self.multiscale.multiscale_transfer()
    #     structural_force = self.multiscale.get_dV()

    #     modal_forcing = self.genericmodalmodel.calculate_modal_forcing(structural_force)
    #     config.logger.info("Modal Forcing: {}".format(modal_forcing[0]))
    #     displacements = self.genericmodalmodel.calculate_modal_displacements(modal_forcing, 0.5)
    #     self.multiscale2.multiscale_solve(displacements)
    #     self.multiscale2.multiscale_transfer()
    #     aero_displacements = self.multiscale2.get_dV()


    #     aero_disp = list(aero_displacements.flatten())
    #     u = aero_disp

    # u = MPI.COMM_WORLD.bcast(u, root=0)
    # dt = MPI.COMM_WORLD.bcast(dt, root=0)
    # # Perform RBF and mesh updates
    # self.fsi.deform_mesh(self.mesh[0], u, dt, False)
    pass


def post_advance(self):
    if self.total_cycles % self.parameters['fsi']['user variables']['fsi frequency'] == 0:
        # initialise displacement list
        u = [0.0 * ii for ii in range(self.num_nodes * 3)]
        pressure_force = get_pressure_force(self)
        dt = self.real_time_step
        dt = 1
        if self.rank == 0:
            # rank 0 tasks
            config.logger.info("Pressure forces: {}".format(np.sum(pressure_force, axis=0)))
            self.multiscale.multiscale_solve(pressure_force)
            self.multiscale.multiscale_transfer()
            structural_force = self.multiscale.get_dV()

            modal_forcing = self.genericmodalmodel.calculate_modal_forcing(structural_force)
            config.logger.info("Structural forces: {}".format(np.sum(structural_force, axis=0)))
            config.logger.info("Modal Forcing: {}".format(modal_forcing[0]))
            print(modal_forcing)
            displacements = self.genericmodalmodel.calculate_modal_displacements(modal_forcing, 0.5)
            self.multiscale2.multiscale_solve(displacements)
            self.multiscale2.multiscale_transfer()
            aero_displacements = self.multiscale2.get_dV()


            aero_disp = list(aero_displacements.flatten())
            u = aero_disp

        u = MPI.COMM_WORLD.bcast(u, root=0)
        dt = MPI.COMM_WORLD.bcast(dt, root=0)
        # Perform RBF and mesh updates
        print(max(u))
        self.fsi.deform_mesh(self.mesh[0], u, dt, False)


def post_solve(self):
    if self.rank == 0:
        # rank 0 tasks
        pass


generic_fsi_coupling = {'post init hook': post_init,
                        'start real time cycle hook': start_real_time_cycle,
                        'post advance hook': post_advance,
                        'post solve hook': post_solve}
