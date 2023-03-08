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
        self.solverlib, self.parameters['case name'], self.parameters['problem name'], self.parameters)
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
            mm = modeReader.NastranReader(
                self.parameters['fsi']['user variables']['filename'])
            mm.read_grid_points()
            mm.read_modes()
        elif self.parameters['fsi']['user variables']['filetype'] == 'cba':
            mm = modeReader.cba_modal(
                self.parameters["fsi"]["user variables"]["filename"])
            mm.calculate_mode_frequencies()

        # create native modal solver
        self.genericmodalmodel = genericmodalmodel(
            mm.grid_points, mm.mode_freqs, mm.mode_data, self.real_time_step, integrator=self.parameters['fsi']['user variables']['integrator'])

        if 'initial forcing' in self.parameters['fsi']['user variables']['']
        # intitial_displacements = self.genericmodalmodel.set_initial_conditions(
        #     [0 for i in range(mm.num_modes)])

        # if (self.parameters['fsi']['user variables']['initial forcing']):
        #     intitial_displacements = self.genericmodalmodel.set_initial_conditions(
        #         self.parameters['fsi']['user variables']['initial forcing'])

        # if (self.parameters['fsi']['user variables']['initial forcing']):
        #     intitial_displacements = self.genericmodalmodel.set_initial_conditions(
        #         self.parameters['fsi']['user variables']['initial forcing'])

        # create data interpolators

        self.IDWMapper = IDWMapper(
            self.aero_centres, mm.grid_points)

        self.multiscaleInterpolator = multiscale(
            self.genericmodalmodel.grid_points, self.parameters['fsi']['base point fraction'], self.parameters['fsi']['alpha'] * 10, False)
        self.multiscaleInterpolator.sample_control_points(False)
        self.multiscaleInterpolator.preprocess_V(self.aero_nodes)

        # self.genericmodalmodel.write_deformed_csv(
        #     intitial_displacements, int(self.real_time_cycle))
        # self.genericmodalmodel.write_grid_csv()

        # self.multiscaleInterpolator.multiscale_solve(
        #     intitial_displacements, False)
        # self.multiscaleInterpolator.multiscale_transfer()

        # self.aero_displacements = self.multiscaleInterpolator.get_dV()
        # self.aero_displacementsTn = self.aero_displacements.copy()

        # self.aero_displacements[np.abs(self.aero_displacements) < 1e-10] = 0

        # with open("displacements_{:04d}.csv".format(int(self.real_time_cycle)), 'w') as f:
        #     f.write("X, Y, Z\n")
        #     for i in range(self.num_nodes):
        #         for j in range(3):
        #             f.write('{}, '.format(
        #                 self.aero_nodes[i, j] + self.aero_displacements[i, j]))
        #         f.write('\n')

        # aero_disp = list(self.aero_displacements.flatten())
        # u = aero_disp
    


    MPI.COMM_WORLD.Barrier()

    # RBF pre-processing
    self.fsi.init_morphing(self.mesh[0])

    # dt = self.real_time_step

    # u = MPI.COMM_WORLD.bcast(u, root=0)
    # dt = MPI.COMM_WORLD.bcast(dt, root=0)

    # Perform RBF and mesh updates
    # config.logger.info("max U: {}".format(max(u)))
    # self.fsi.deform_mesh(self.mesh[0], u, dt, False)
    self.pseudo_fsi_cycles = 0


def start_real_time_cycle(self):
    if (self.total_cycles % self.parameters['fsi']['user variables']['fsi frequency'] == 0) and (self.total_cycles != 0) and (self.real_time_cycle >= self.parameters['fsi']['user variables']['time threshold']):
        # initialise displacement list
        u = [0.0 * ii for ii in range(self.num_nodes * 3)]
        pressure_force = get_pressure_force(self) * self.fsi_scaling
        dt = self.real_time_step

        if self.rank == 0:
            # rank 0 tasks

            # Calculate forces on structural model
            structural_force = self.IDWMapper.map(pressure_force, n=100)

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
            displacements = self.genericmodalmodel.get_displacements() / 1000

            self.genericmodalmodel.write_force_history(
                self.real_time_cycle, self.real_time_cycle * dt)

            self.genericmodalmodel.write_deformed_csv(
                displacements, int(self.real_time_cycle))
            self.genericmodalmodel.write_grid_csv()

            self.multiscaleInterpolator.multiscale_solve(displacements, False)
            self.multiscaleInterpolator.multiscale_transfer()
            self.aero_displacements = self.multiscaleInterpolator.get_dV()

            with open("displacements_{:04d}.csv".format(int(self.real_time_cycle)), 'w') as f:
                f.write("X, Y, Z\n")
                for i in range(self.num_nodes):
                    for j in range(3):
                        f.write('{}, '.format(
                            self.aero_nodes[i, j] + self.aero_displacements[i, j]))
                    f.write('\n')

            # Check conservation of virtual work

            # structureToCentres = multiscale(self.genericmodalmodel.grid_points, self.parameters['fsi']['base point fraction'], self.parameters['fsi']['alpha'])
            # structureToCentres.sample_control_points(False)
            # structureToCentres.preprocess_V(self.aero_centres)

            # structureToCentres.multiscale_solve(displacements, False)
            # structureToCentres.multiscale_transfer()
            # centre_displacements = structureToCentres.get_dV()

            # structural_virtual_work = np.multiply(structural_force, displacements)
            # aero_virtual_work = np.multiply(centre_displacements, pressure_force)

            # config.logger.info("Checking conservation of virtual work: ")
            # config.logger.info("Structural VW: {}".format(np.sum(structural_virtual_work, axis=0)))
            # config.logger.info("Aero VW: {}".format(np.sum(aero_virtual_work, axis=0)))

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
        self.fsi.deform_mesh(self.mesh[0], u, dt, False)
        self.pseudo_fsi_cycles = 0


def post_advance(self):
    self.pseudo_fsi_cycles += 1
    if (self.total_cycles % self.parameters['fsi']['user variables']['pseudo fsi frequency'] == 0) and (self.total_cycles != 0) and (self.real_time_cycle >= self.parameters['fsi']['user variables']['time threshold']) and (self.pseudo_fsi_cycles >= self.parameters['fsi']['user variables']['pseudo time threshold']):
        # initialise displacement list
        u = [0.0 * ii for ii in range(self.num_nodes * 3)]
        pressure_force = get_pressure_force(self) * self.fsi_scaling
        dt = self.real_time_step

        if self.rank == 0:
            # rank 0 tasks

            # Calculate forces on structural model
            structural_force = self.IDWMapper.map(pressure_force, n=10)

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

            self.multiscaleInterpolator.multiscale_solve(displacements, False)
            self.multiscaleInterpolator.multiscale_transfer()
            self.aero_displacements = self.multiscaleInterpolator.get_dV()

            with open("displacements_{:04d}.csv".format(int(self.real_time_cycle)), 'w') as f:
                f.write("X, Y, Z\n")
                for i in range(self.num_nodes):
                    for j in range(3):
                        f.write('{}, '.format(
                            self.aero_nodes[i, j] + self.aero_displacements[i, j]))
                    f.write('\n')

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
        self.fsi.deform_mesh(self.mesh[0], u, dt, False)


def post_solve(self):
    if self.rank == 0:
        # rank 0 tasks
        pass


generic_fsi_coupling = {'post init hook': post_init,
                        'start real time cycle hook': start_real_time_cycle,
                        'post advance hook': post_advance,
                        'post solve hook': post_solve}
