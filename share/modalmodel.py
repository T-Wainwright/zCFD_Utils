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

import importlib
from mpi4py import MPI
from zcfd.solvers.utils.RuntimeLoader import create_modal_model, create_mapper
from zcfd.utils import config
# from zcfd.utils.NastranReader import NastranReader
from zcfd.utils.GenericModeReader import GenericModeReader


def post_init(self):
    self.modal_models = {}
    self.mappers = {}
    mm_update = []
    rank = MPI.COMM_WORLD.Get_rank()

    import importlib

    multiscaleLib = importlib.import_module("libmultiscale.so")

    model = getattr(multiscaleLib, "Multiscale")

    if "modal model" in self.parameters:
        module_suffix = '_INTEL'
        if config.options.device == "gpu":
            module_suffix = '_CUDA'

        mapperlib = importlib.import_module("libzCFDMapper" + module_suffix)
        for k in self.parameters['modal model']:
            if 'update frequency' in self.parameters["modal model"][k]:
                mm_update.append(
                    self.parameters["modal model"][k]["update frequency"])

            self.modal_models[k] = create_modal_model(
                self.solverlib, self.parameters['case name'], self.parameters['problem name'], k.upper(), self.parameters)
            self.mappers[k] = create_mapper(mapperlib, [
                                            True] * MPI.COMM_WORLD.Get_size(), [True] * MPI.COMM_WORLD.Get_size(), self.parameters)

            if self.real_time_step == self.total_time:
                self.steady_fsi = True
                self.modal_models[k].set_steady_state()
                self.fsi_time_step = 1.0
            else:
                self.steady_fsi = False
                self.fsi_time_step = self.real_time_step

            # nastran_reader = NastranReader(self.parameters["modal model"][k]["nastran casename"], self.modal_models[k])
            genericModeReader = GenericModeReader(
                '/home/tom.wainwright/cases/MDO_250K/CBA_Modes_Coincident.h5', self.modal_models[k])
            # nastran_reader.read_grid_points()
            mode_list = []
            if 'mode list' in self.parameters["modal model"][k]:
                mode_list = self.parameters["modal model"][k]["mode list"]
            modal_damping = []
            if 'modal damping' in self.parameters["modal model"][k]:
                modal_damping = self.parameters["modal model"][k]["modal damping"]
            genericModeReader.add_modes(mode_list, modal_damping, self.mesh[0])
            self.modal_models[k].init_zones_and_storage(self.mesh[0])

            if rank == 0:
                self.mappers[k].map_structural_points(self.modal_models[k].get_grid_node_array(),
                                                      self.mesh[0].get_face_centres(
                ),
                    self.modal_models[k].get_mesh_face_centre_indexes(
                ),
                    k.upper()
                )
            else:
                self.mappers[k].map_structural_points(None,
                                                      self.mesh[0].get_face_centres(
                                                      ),
                                                      self.modal_models[k].get_mesh_face_centre_indexes(
                                                      ),
                                                      k.upper()
                                                      )

            for ii in range(genericModeReader.num_modes):
                self.mappers[k].map_data(4, self.modal_models[k].get_mode_shape_data(
                    ii), self.modal_models[k].get_boundary_mode_shape_data_on_faces(ii), False)

            if rank == 0:
                self.mappers[k].map_structural_points(self.modal_models[k].get_grid_node_array(),
                                                      self.mesh[0].get_node_vertices(
                ),
                    self.modal_models[k].get_mesh_node_indexes(
                ),
                    k.upper()
                )
            else:
                self.mappers[k].map_structural_points(None,
                                                      self.mesh[0].get_node_vertices(
                                                      ),
                                                      self.modal_models[k].get_mesh_node_indexes(
                                                      ),
                                                      k.upper()
                                                      )

            for ii in range(genericModeReader.num_modes):
                self.mappers[k].map_data(4, self.modal_models[k].get_mode_shape_data(
                    ii), self.modal_models[k].get_boundary_mode_shape_data_on_nodes(ii), False)

            self.solver.add_modal_model(self.modal_models[k])
            self.modal_models[k].init_morphing(self.mesh[0])

        for mm in list(self.modal_models.keys()):
            config.logger.info("restart {}".format(self.parameters['restart']))
            if self.parameters['restart']:
                self.modal_models[mm].read_results(
                    self.parameters['case name'])
                self.modal_models[mm].deform_mesh(
                    self.mesh[0], self.fsi_time_step, False)
            # else:
            self.modal_models[mm].copy_time_history(self.mesh[0])

    self.mm_update = 1
    if mm_update != []:
        new_update = 100000
        for u in mm_update:
            new_update = min(u, new_update)

        self.mm_update = new_update

    config.logger.info(
        " Modal model updating every {} cycles".format(self.mm_update))


def start_real_time_cycle(self):
    if self.local_timestepping:
        self.solver.calculate_modal_forcing(
            self.fsi_time_step, self.real_time_cycle * self.fsi_time_step)
        for mm in list(self.modal_models.keys()):
            self.modal_models[mm].copy_time_history(self.mesh[0])
            self.modal_models[mm].copy_force_history()


def post_advance(self):
    if self.solve_cycle % self.mm_update == 0:
        current_time = (self.solve_cycle - 1) * self.real_time_step
        if self.local_timestepping:
            current_time = self.real_time_cycle * self.real_time_step

        self.solver.calculate_modal_forcing(self.fsi_time_step, current_time)
        for mm in list(self.modal_models.keys()):
            if self.steady_fsi:
                self.modal_models[mm].march(
                    self.solve_cycle, self.fsi_time_step)
                self.modal_models[mm].copy_time_history(self.mesh[0])
            else:
                self.modal_models[mm].march(current_time, self.fsi_time_step)

            self.modal_models[mm].deform_mesh(self.mesh[0], self.fsi_time_step)

            if not self.local_timestepping:
                self.modal_models[mm].copy_time_history(self.mesh[0])
                self.modal_models[mm].copy_force_history()

        self.solver.update_halos(False)

        config.cycle_info = self.solver.init_solution(
            self.parameters['case name'])


def post_output(self):
    if self.should_output():
        current_time = (self.solve_cycle - 1) * self.real_time_step
        if self.local_timestepping:
            current_time = self.real_time_cycle * self.real_time_step

        for mm in list(self.modal_models.keys()):
            self.modal_models[mm].write_results(
                self.parameters['case name'], self.real_time_cycle, self.solve_cycle, current_time)


modal_model_coupling = {'post init hook': post_init,
                        'start real time cycle hook': start_real_time_cycle,
                        'post advance hook': post_advance,
                        'post output hook': post_output}
