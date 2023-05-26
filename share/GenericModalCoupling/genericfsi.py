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
from libmultiscale import multiscale

from zcfdutils.genericmodalmodel import genericmodalmodel
from zcfdutils.py_rbf import IDWMapper
from zcfdutils.Wrappers import modeReader


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
    # create solver interface object
    self.fsi = create_generic_fsi(
        self.solverlib, self.parameters['case name'], self.parameters['mesh name'], self.parameters)
    self.fsi.init(self.mesh[0])

    # Get rank
    self.rank = MPI.COMM_WORLD.Get_rank()

    # Get nodes on FSI surface
    flat_nodes = self.fsi.get_fsi_nodes(self.mesh[0])

    self.num_nodes = int(len(flat_nodes) / 3)
    flat_centres = self.fsi.get_faceCentres(self.mesh[0])

    self.aero_centres = np.reshape(
        flat_centres, (int(len(flat_centres) / 3), 3))
    self.aero_nodes = np.reshape(flat_nodes, ((self.num_nodes, 3)))

    self.fsi_cycles = 0

    self.fsi_scaling = self.parameters['fsi']['user variables']['fluid force scaling']

    u = [0.0 * ii for ii in range(self.num_nodes * 3)]

    self.aero_displacements = np.zeros_like(self.aero_nodes)
    self.aero_displacementsTn = np.zeros_like(self.aero_nodes)

    # Rank 0 tasks
    if self.rank == 0:
        # use modal reader to read in structural model

        # select reader
        if self.parameters['fsi']['user variables']['filetype'] == 'nastran':
            self.mm = modeReader.NastranReader(
                self.parameters['fsi']['user variables']['filename'])
            self.mm.read_grid_points()
            self.mm.read_modes()
        elif self.parameters['fsi']['user variables']['filetype'] == 'cba':
            self.mm = modeReader.cba_modal(
                self.parameters["fsi"]["user variables"]["filename"])
            self.mm.calculate_mode_frequencies()
        elif self.parameters['fsi']['user variables']['filetype'] == 'atom':
            self.mm = modeReader.atomReader(self.parameters['fsi']['user variables']['filename'])

        # create native modal solver
        self.genericmodalmodel = genericmodalmodel(self.mm, self.real_time_step, integrator=self.parameters['fsi']['user variables']['integrator'])

        self.IDWMapper = IDWMapper(
            self.aero_centres, self.mm.get_loading_nodes())

        config.logger.info("moving nodes shape: {}".format(self.mm.get_moving_nodes().shape))

        self.multiscaleInterpolator = multiscale(
            self.mm.get_moving_nodes(), 1.0, 120.0, True)
        self.multiscaleInterpolator.sample_control_points(False)
        self.multiscaleInterpolator.preprocess_V(self.aero_nodes)


    MPI.COMM_WORLD.Barrier()

    # RBF pre-processing
    self.fsi.init_morphing(self.mesh[0])

    self.pseudo_fsi_cycles = 0


def start_real_time_cycle(self):
    if (self.total_cycles % self.parameters['fsi']['user variables']['fsi frequency'] == 0) and (self.total_cycles != 0) and (self.real_time_cycle >= self.parameters['fsi']['user variables']['time threshold']):
        # initialise displacement list
        u = [0.0 * ii for ii in range(self.num_nodes * 3)]
        pressure_force = get_pressure_force(self) * self.fsi_scaling
        dt = self.real_time_step
        dt = 1.0

        if self.rank == 0:
            # rank 0 tasks

            # Calculate forces on structural model
            structural_force = self.IDWMapper.map(pressure_force, n=5)

            # Check conservation of forces

            config.logger.info("Pressure forces: {}".format(
                np.sum(pressure_force, axis=0)))
            config.logger.info("Checking conservation of forces: ")
            config.logger.info("Structural forces: {}".format(
                np.sum(structural_force, axis=0)))

            # Calculate modal forcing
            modal_forcing = self.genericmodalmodel.calculate_modal_forcing(
                structural_force)

            self.genericmodalmodel.integrate_solution()
            displacements = self.genericmodalmodel.get_displacements()
            self.genericmodalmodel.write_deformed_csv(
                displacements, int(self.real_time_cycle))
            self.genericmodalmodel.write_grid_csv()

            if self.parameters['fsi']['user variables']['filetype'] == 'atom':
                translation, rotation = self.mm.deform_ribs(displacements)
                rib_deformations = translation + rotation
                self.mm.write_beamstick_deformed(rib_deformations)
                displacements = np.concatenate((displacements[:, 0:3], rib_deformations), axis=0)

            self.genericmodalmodel.write_force_history(
                self.real_time_cycle, self.real_time_cycle * dt)

            self.multiscaleInterpolator.multiscale_solve(displacements, False)
            self.multiscaleInterpolator.multiscale_transfer()
            self.aero_displacements = self.multiscaleInterpolator.get_dV()

            # k = 2/(120 ** 2)
            # self.aero_displacements = np.zeros_like(self.aero_displacements)
            # self.aero_displacements[:, 0] = self.aero_nodes[:, 2]**2 * k


            with open("displacements_{:04d}.csv".format(int(self.real_time_cycle)), 'w') as f:
                f.write("X, Y, Z\n")
                for i in range(self.num_nodes):
                    for j in range(3):
                        f.write('{}, '.format(
                            self.aero_nodes[i, j] + self.aero_displacements[i, j]))
                    f.write('\n')

            config.logger.info("Displacement range: ")
            config.logger.info("{}".format(np.min(self.aero_displacements, axis=0)))
            config.logger.info("{}".format(np.max(self.aero_displacements, axis=0)))


            delta_u = self.aero_displacements

            aero_displacements = list(delta_u.flatten())
            u = aero_displacements

            config.logger.info("min abs(u): {}".format(np.min(np.abs(u))))
            config.logger.info("max abs(u): {}".format(np.max(np.abs(u))))



            self.aero_displacementsTn = self.aero_displacements.copy()

        u = MPI.COMM_WORLD.bcast(u, root=0)
        dt = MPI.COMM_WORLD.bcast(dt, root=0)
        # Perform RBF and mesh updates
        config.logger.info("max U: {}".format(max(u)))
        self.fsi.deform_mesh(self.mesh[0], u, dt, True)
        self.pseudo_fsi_cycles = 0

        # clear out more meshes

        self.solver.generate_coarse_meshes(self.mesh)



def post_advance(self):
    self.pseudo_fsi_cycles += 1
    if (self.total_cycles % self.parameters['fsi']['user variables']['pseudo fsi frequency'] == 0) and (self.total_cycles != 0) and (self.real_time_cycle >= self.parameters['fsi']['user variables']['time threshold']) and (self.pseudo_fsi_cycles >= self.parameters['fsi']['user variables']['pseudo time threshold']):
        # initialise displacement list
        u = [0.0 * ii for ii in range(self.num_nodes * 3)]
        pressure_force = get_pressure_force(self) * self.fsi_scaling
        dt = self.real_time_step
        dt = 1.0

        if self.rank == 0:
            # rank 0 tasks

            # Calculate forces on structural model
            structural_force = self.IDWMapper.map(pressure_force, n=5)

            # Check conservation of forces

            config.logger.info("Pressure forces: {}".format(
                np.sum(pressure_force, axis=0)))
            config.logger.info("Checking conservation of forces: ")
            config.logger.info("Structural forces: {}".format(
                np.sum(structural_force, axis=0)))

            # Calculate modal forcing
            modal_forcing = self.genericmodalmodel.calculate_modal_forcing(
                structural_force)

            self.genericmodalmodel.integrate_solution()
            displacements = self.genericmodalmodel.get_displacements() * self.fsi_scaling

            self.genericmodalmodel.write_force_history(
                self.real_time_cycle, self.real_time_cycle * dt)

            self.genericmodalmodel.write_deformed_csv(
                displacements, int(self.real_time_cycle))
            self.genericmodalmodel.write_grid_csv()

            if self.parameters['fsi']['user variables']['filetype'] == 'atom':
                translation, rotation = self.mm.deform_ribs(displacements)
                rib_deformations = translation + rotation
                self.mm.write_beamstick_deformed(rib_deformations)
                displacements = np.concatenate((displacements[:, 0:3], rib_deformations), axis=0)

            self.genericmodalmodel.write_force_history(
                self.real_time_cycle, self.real_time_cycle * dt)
        

            self.multiscaleInterpolator.multiscale_solve(displacements, False)
            self.multiscaleInterpolator.multiscale_transfer()
            self.aero_displacements = self.multiscaleInterpolator.get_dV()

            # k = 2/(120 ** 2)
            # self.aero_displacements = np.zeros_like(self.aero_displacements)
            # self.aero_displacements[:, 0] = self.aero_nodes[:, 2]**2 * k

            with open("displacements_{:04d}.csv".format(int(self.real_time_cycle)), 'w') as f:
                f.write("X, Y, Z\n")
                for i in range(self.num_nodes):
                    for j in range(3):
                        f.write('{}, '.format(
                            self.aero_nodes[i, j] + self.aero_displacements[i, j]))
                    f.write('\n')

            
            config.logger.info("Displacement range: ")
            config.logger.info("{}".format(np.min(self.aero_displacements, axis=0)))
            config.logger.info("{}".format(np.max(self.aero_displacements, axis=0)))


            delta_u = self.aero_displacements

            aero_disp = list(delta_u.flatten())
            u = aero_disp
            config.logger.info("min abs(u): {}".format(np.min(np.abs(u))))
            config.logger.info("max abs(u): {}".format(np.max(np.abs(u))))

            self.aero_displacementsTn = self.aero_displacements.copy()

        u = MPI.COMM_WORLD.bcast(u, root=0)
        dt = MPI.COMM_WORLD.bcast(dt, root=0)
        # Perform RBF and mesh updates
        config.logger.info("max U: {}".format(max(u)))
        self.fsi.deform_mesh(self.mesh[0], u, dt, True)

        self.solver.generate_coarse_meshes(self.mesh)


def post_solve(self):
    if self.rank == 0:
        # rank 0 tasks
        pass


generic_fsi_coupling = {'post init hook': post_init,
                        'start real time cycle hook': start_real_time_cycle,
                        'post advance hook': post_advance,
                        'post solve hook': post_solve}
