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
from zcfd.solvers.utils.RuntimeLoader import create_abaqus_fsi
from zcfd.utils import config
from zcfd.utils.coupling import py_rbf
import sys


# import libsimulia_cse as simulia_cse


def post_init(self):
    self.abaqus_fsi = create_abaqus_fsi(self.solverlib, self.parameters['case name'], self.parameters['problem name'], self.parameters)

    num_faces = self.abaqus_fsi.init(self.mesh[0])

    p = self.abaqus_fsi.get_pressures(self.solver_data[0], self.mesh[0])
    nodes = self.abaqus_fsi.get_fsi_nodes(self.mesh[0])  # x, y, z of nodes on fsi surface
    face_nodes = self.abaqus_fsi.get_fsi_face_nodes()  # face nodes ALL QUADS CURRENTLY
    self.num_nodes = int(len(nodes) / 3)

    self.rank = MPI.COMM_WORLD.Get_rank()

    # Rank 0 stuff
    if self.rank == 0:
        self.node_labels = self.abaqus_fsi.get_fsi_node_labels()
        # Aero-preprossessing
        self.UoB_coupling = py_rbf.UoB_coupling()
        self.UoB_coupling.aero_nodes = np.reshape(nodes, [self.num_nodes, 3])
        self.UoB_coupling.n_a = self.num_nodes
        self.UoB_coupling.n_f = num_faces
        self.UoB_coupling.face_nodes = face_nodes
        self.UoB_coupling.node_labels = self.node_labels
        self.UoB_coupling.process_face()

        # load structural nodes
        self.UoB_coupling.load_struct('beamstick.dat')
        self.UoB_coupling.load_modes()

        self.UoB_coupling.generate_displacement_transfer_matrix(self.parameters['abaqus fsi']['alpha'], polynomial=False)
        self.UoB_coupling.generate_pressure_transfer_matrix(self.parameters['abaqus fsi']['alpha'], polynomial=False)

    MPI.COMM_WORLD.Barrier()

    # RBF pre-processing
    self.abaqus_fsi.init_morphing(self.mesh[0])

    print('Finished setup')


def start_real_time_cycle(self):
    U_a = [0.0 * ii for ii in range(self.num_nodes * 3)]
    dt = self.real_time_step

    p = self.abaqus_fsi.get_pressures(self.solver_data[0], self.mesh[0])

    if self.rank == 0:
        for ii in range(len(p)):
            p[ii] -= self.parameters['ic_2']['pressure']
            # print("ii {} pressure {}".format(ii, p[ii]), flush=True)

        F_a = self.UoB_coupling.calculate_pressure_force(p)
        F_s = self.UoB_coupling.interp_forces(F_a)

        U_s = self.UoB_coupling.deform_struct(F_s)
        U_s = U_s[:, 0:3]
        U_a = self.UoB_coupling.interp_displacements(U_s)
        U_a = list(U_a.flatten(order='C'))

        max_disp = 0.0
        for ii in range(self.num_nodes):
            idx = ii * 3
            disp = np.sqrt(U_a[idx] * U_a[idx] + U_a[idx + 1] * U_a[idx + 1] + U_a[idx + 2] * U_a[idx + 2])
            max_disp = max(max_disp, disp)

        print("max displacement {}".format(max_disp), flush=True)
        # delta  = -40.0 * float(self.real_time_step) * float(self.solve_cycle)
        # for ii in range(self.num_nodes):
        #     u[ii*3+2] = delta

        # print("deformation dist {}".format(delta), flush=True)

    # sys.exit()
    # if self.rank == 0:
    #     lockstep = 1
    #     controls = 0
    #     current_time = self.real_time_cycle * self.real_time_step if self.local_timestepping else (self.solve_cycle - 1) * self.real_time_step
    #     target_time = (self.real_time_cycle + 1) * self.real_time_step if self.local_timestepping else (self.solve_cycle) * self.real_time_step

    #     time_data = self.cse.get_target_time(current_time, self.real_time_step, target_time, self.real_time_step, lockstep, controls)
    #     target_time = time_data[0]
    #     dt = time_data[1]
    #     lockstep = time_data[2]
    #     controls = time_data[3]

    #     self.cse.notify_start()

    #     self.cse.get_field("displacement", "zCFD Mesh", "NULL", self.num_nodes, self.node_labels, target_time, u)

    U_a = MPI.COMM_WORLD.bcast(U_a, root=0)
    dt = MPI.COMM_WORLD.bcast(dt, root=0)
    # Perform RBF and mesh updates
    self.abaqus_fsi.deform_mesh(self.mesh[0], U_a, dt, False)


def post_advance(self):
    if self.solve_cycle % 10 == 0 and self.solve_cycle > 0 and self.real_time_step > 100.0:
        U_a = [0.0 * ii for ii in range(self.num_nodes * 3)]
        dt = self.real_time_step
        p = self.abaqus_fsi.get_pressures(self.solver_data[0], self.mesh[0])

        if self.rank == 0:
            print('Solving FSI')
            for ii in range(len(p)):
                p[ii] -= self.parameters['ic_2']['pressure']

            F_a = self.UoB_coupling.calculate_pressure_force(p)
            print("Aero force summation: {} \t {} \t {}".format(np.sum(F_a[:, 0]), np.sum(F_a[:, 1]), np.sum(F_a[:, 2])), flush=True)
            F_s = self.UoB_coupling.interp_forces(F_a)
            print("Struct force summation: {} \t {} \t {}".format(np.sum(F_s[:, 0]), np.sum(F_s[:, 1]), np.sum(F_s[:, 2])), flush=True)

            print('Deforming structural nodes')
            U_s = self.UoB_coupling.deform_struct(F_s)
            U_s = U_s[:, 0:3]

            print('Interpolating aero surface displacements')
            U_a = self.UoB_coupling.interp_displacements(U_s)
            print('Updating surface normals')
            self.UoB_coupling.update_surface(U_a)

            U_a = list(U_a.flatten(order='C'))

            max_disp = 0.0
            for ii in range(self.num_nodes):
                idx = ii * 3
                disp = np.sqrt(U_a[idx] * U_a[idx] + U_a[idx + 1] * U_a[idx + 1] + U_a[idx + 2] * U_a[idx + 2])
                max_disp = max(max_disp, disp)

            print("max displacement {}".format(max_disp), flush=True)
            print('Displacing volume mesh')

            self.UoB_coupling.write_deformed_struct(U_s)

        U_a = MPI.COMM_WORLD.bcast(U_a, root=0)
        dt = MPI.COMM_WORLD.bcast(dt, root=0)
        # Perform RBF and mesh updates
        self.abaqus_fsi.deform_mesh(self.mesh[0], U_a, dt, False)

    config.logger.info("finished post advance")
    # p = self.abaqus_fsi.get_pressures(self.solver_data[0], self.mesh[0])
    # if self.rank == 0:
    #     self.cse.set_pressure(p)
    #     t = (self.real_time_cycle + 1) * self.real_time_step if self.local_timestepping else (self.solve_cycle) * self.real_time_step

#        self.cse.notify_end(t)


def post_output(self):
    # Output to results.h5
    pass


abaqus_fsi_coupling = {'post init hook': post_init,
                       'start real time cycle hook': start_real_time_cycle,
                       'post advance hook': post_advance,
                       'post output hook': post_output}
