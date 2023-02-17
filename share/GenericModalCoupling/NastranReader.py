"""
Copyright (c) 2012-2019, Zenotech Ltd
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

import os
import numpy as np
from zcfd.utils import config
import pyNastran
from pyNastran.op2.op2 import read_op2
from pyNastran.bdf.bdf import BDF
from mpi4py import MPI


class NastranReader:
    def __init__(self, filename, modal_model=None):
        self.modal_model = modal_model
        self.filename = filename
        config.logger.info(" Reading Nastran data from {}".format(self.filename))
        self.op2 = read_op2(filename + ".op2", build_dataframe=False, debug=False)
        self.bdf = BDF(debug=False)

    def read_grid_points(self):
        rank = MPI.COMM_WORLD.Get_rank()
        if rank == 0:
            config.logger.info(" Reading grid points from %s" % self.filename + ".bdf")
            self.bdf.read_bdf(self.filename + ".bdf")

            count = 0
            self.structural_nodes = set()

            mass_elements = ['CONM1', 'CONM2', 'CMASS1', 'CMASS2', 'CMASS3', 'CMASS4']

            for _, elem in self.bdf.elements.items():
                if elem.type not in mass_elements:
                    self.structural_nodes.update(elem.node_ids)

            self.num_grid_points = len(self.structural_nodes)
            grid_points = np.zeros((self.num_grid_points, 3))

            for node_id in self.structural_nodes:
                node = self.bdf.nodes[node_id]
                for ii, loc in enumerate(node.xyz):
                    grid_points[count][ii] = float(loc)

                count += 1

            config.logger.info(" Read %i grid points" % self.num_grid_points)

            if self.modal_model:
                self.modal_model.add_grid_points(grid_points)

            self.grid_points = grid_points

    def read_modes(self, mode_list, modal_damping, mesh):

        num_cases = len(list(self.op2.eigenvectors.keys()))
        config.logger.info(" Reading %i loadcase(s) from %s" % (num_cases, self.filename + ".op2"))

        if num_cases > 1:
            config.logger.error(" More than one load case in %s" % self.filename)

        eig1 = self.op2.eigenvectors[1]
        self.mode_freqs = list(eig1.mode_cycles)
        for i in range(len(self.mode_freqs)):
            self.mode_freqs[i] *= 2.0 * np.pi
        self.eigenvals = eig1.eigns
        self.num_modes = len(eig1.modes)

        # Mode selection
        if len(mode_list) == 0:
            mode_list = [ii for ii in range(len(self.num_modes))]
        else:
            self.num_modes = len(mode_list)

        node_id_to_index = {}
        for i, node_id in enumerate(self.bdf.nodes.keys()):
            node_id_to_index[node_id] = i

        rank = MPI.COMM_WORLD.Get_rank()
        if rank == 0:
            mode_data = np.zeros((self.num_modes, self.num_grid_points, 6))
            for ii in range(self.num_modes):
                for jj, node_id in enumerate(self.structural_nodes):
                    op2_index = node_id_to_index[node_id]
                    mode_data[ii, jj, :] = eig1.data[mode_list[ii], op2_index, :]
        else:
            mode_data = np.empty((1, 0))



        self.mode_freqs = [self.mode_freqs[i] for i in mode_list]

        # Modal damping
        if len(modal_damping) > 0:
            self.mode_damping = modal_damping
        else:
            self.mode_damping = [0.0] * self.num_modes

        config.logger.info(" Read %i modes" % self.num_modes)
        config.logger.info(" Modal frequencies (Hz) %s" % " ".join(['{: .2f}'.format(freq / (2.0 * np.pi)) for freq in self.mode_freqs]))

        if self.modal_model:
            self.modal_model.add_modes(mode_data, self.mode_freqs, self.mode_damping, self.num_modes, mesh)

        self.mode_data = mode_data
        self.mode_list = mode_list
        self.modal_displacements = [0 for i in range(self.num_modes)]

    def print_op2_stats(self, short=False):
        rank = MPI.COMM_WORLD.Get_rank()
        if rank == 0:
            print(self.op2.get_op2_stats(short), flush=True)

    def print_bdf(self):
        rank = MPI.COMM_WORLD.Get_rank()
        if rank == 0:
            print(self.bdf.object_attributes(), flush=True)
            print(self.bdf.object_methods(), flush=True)

    def calculate_modal_forcing(self, force):
        modal_forcing = [0 for i in range(self.num_modes)]
        for m in range(self.num_modes):
            for i in range(self.num_grid_points):
                for k in range(3):
                    modal_forcing[m] += self.mode_data[m, i, k] * force[k]
        return modal_forcing

    def calculate_modal_displacements(self, modal_forcing, relaxation_factor):
        displacements = np.zeros_like(self.grid_points)
        for m in range(self.num_modes):
            self.modal_displacements = self.modal_displacements + relaxation_factor * (modal_forcing[m] - self.mode_freqs[m] ** 2 * self.modal_displacements[m])
            for i in range(self.num_grid_points):
                for j in range(3):
                    displacements[i, j] += self.modal_displacements[m] * self.mode_data[m, i, j]
        return displacements

            


