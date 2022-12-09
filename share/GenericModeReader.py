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
import h5py
from mpi4py import MPI


class GenericModeReader:
    def __init__(self, filename, modal_model=None):
        self.modal_model = modal_model
        self.filename = filename
        config.logger.info(" Reading Mode data from {}".format(self.filename))
        f = h5py.File(self.filename, "r")
        self.grid_points = np.array(f.get("grid_points"))
        self.mode_frequencies = list(f.get("mode_frequencies"))
        self.mode_data = np.array(f.get("mode_data"))

        self.num_nodes = self.grid_points.shape[0]
        self.num_modes = self.mode_data.shape[0]

        self.mode_damping = [0.0] * self.num_modes

        if self.modal_model:
            self.modal_model.add_grid_points(self.grid_points)

    def add_modes(self, mode_list, modal_damping, mesh):

        if len(mode_list) == 0:
            mode_list = [ii for ii in range(self.num_modes)]
        else:
            self.num_modes = len(mode_list)

        rank = MPI.COMM_WORLD.Get_rank()
        if rank == 0:
            f = h5py.File(self.filename, "r")
            mode_data = np.zeros((self.num_modes, self.num_nodes, 6))
            for ii in range(self.num_modes):
                for jj in range(3):
                    mode_data[ii, :, jj] = self.mode_data[mode_list[ii], :, jj]
        else:
            mode_data = np.empty((1, 0))

        # print(mode_data)

        if len(modal_damping) > 0:
            self.mode_damping = modal_damping
        else:
            self.mode_damping = [0.0] * self.num_modes

        if self.modal_model:
            self.modal_model.add_modes(
                mode_data, self.mode_frequencies, self.mode_damping, self.num_modes, mesh)

        # # dump points file
        # g = open(
        #     "/home/tom/Documents/University/Coding/cases/MDO_250K/mode_grid.csv", "w")
        # g.write("X, Y, Z, mx, my, mz\n")
        # for i in range(self.num_nodes):
        #     g.write('{}, {}, {}, {}, {}, {}\n'.format(
        #         self.grid_points[i, 0], self.grid_points[i, 1], self.grid_points[i, 2], mode_data[0, i, 0], mode_data[0, i, 1], mode_data[0, i, 2]))
        # g.close()
