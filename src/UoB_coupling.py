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

Generic Aero-structural coupling scheme for zcfd

Tom Wainwright
University of Bristol
2020
"""
import numpy as np
import os
import sys
from zcfd import MPI
from zcfd.solvers.utils.RuntimeLoader import create_abaqus_fsi
from zcfd.utils import config
import py_rbf


def post_init(self):
    # Create aero-structural link
    self.abaqus_fsi = create_abaqus_fsi(self.solverlib, 'case_name', 'problem_name')

    # Get face information 
    self.num_faces = self.abaqus_fsi.init(self.mesh[0])
    self.p = self.abaqus_fsi.get_pressures(self.solver_data[0], self.mesh[0])
    self.nodes = self.abaqus_fsi.get_fsi_nodes(self.mesh[0])  # x, y, z of nodes on fsi surface
    self.n_a = len(self.nodes) / 3                     # Number of unique nodes
    self.nodes = np.reshape(self.nodes, [self.n_a, 3])
    self.face_nodes = self.abaqus_fsi.get_fsi_face_nodes()  # face nodes ALL QUADS CURRENTLY
    self.rank = MPI.COMM_WORLD.Get_rank()
    if self.rank == 0:
        self.node_labels = self.abaqus_fsi.get_fsi_node_labels()
    self.pressure_force = np.zeros((self.num_faces, 3))
    self.face_dictionary = process_face(self)


    # read structural nodes - placeholder function for now
    loadbeamstick(self)
    # write_struc_tec(self,'struct.plt')

    write_tec(self,'aero.plt')
    # create coupling matrices- here using two full RBF matrices
    self.H_u = py_rbf.generate_transfer_matrix(self.aero_nodes, self.struct_nodes, self.parameters['abaqus fsi']['alpha'], polynomial=True)
    self.H_p = py_rbf.generate_transfer_matrix(get_face_centres(self), self.struct_nodes, self.parameters['abaqus fsi']['alpha'], polynomial=True)

    # Initiate deformation arrays
    self.U_s = np.zeros((self.n_s, 3))
    self.U_a = np.zeros((self.n_a, 3))

    self.F_s = np.zeros((self.n_s, 3))
    self.F_a = np.zeros((self.n_a, 3))

    # Write initial geometry files
    write_tec(self, 'Original_Surf.plt')
    write_struc_tec(self, 'Original_Beam.plt')

    MPI.COMM_WORLD.Barrier()

    # zCFD preprocessing
    self.abaqus_fsi.init_rbf()


def start_real_time_cycle(self):
    # zero displacement arrays
    self.U_a = np.zeros((self.n_a, 3))
    self.U_s = np.zeros((self.n_s, 3))
    # Read structural displaments
    self.U_s = loadDisplacements(self)
    
    self.U_a = py_rbf.interp_displacements(self.U_s, self.H_u)

    self.nodes[:,0] = self.nodes[:,0] + self.U_a[:, 0]
    self.nodes[:,1] = self.nodes[:,1] + self.U_a[:, 1]
    self.nodes[:,2] = self.nodes[:,2] + self.U_a[:, 2]

    write_tec(self, 'deformed_aero.plt')

    # Deform volume mesh
    dt = self.real_time_step

    self.U_a = MPI.COMM_WORLD.bcast(self.U_a, root=0)
    dt = MPI.COMM_WORLD.bcast(dt, root=0)
    # Perform RBF and mesh updates
    self.abaqus_fsi.deform_mesh(self.mesh[0], list(self.U_a), dt)
    print("Finished start real time cycle")
    sys.exit()


def post_advance(self):
    p = self.abaqus_fsi.get_pressures(self.solver_data[0], self.mesh[0])
    if self.rank == 0:
        t = (self.real_time_cycle + 1) * self.real_time_step if self.local_timestepping else (self.solve_cycle) * self.real_time_step


def post_output(self):
    # Output to results.h5
    pass


# Custom functions -----------------------------------------
# ----------------------------------------------------------
# ----------------------------------------------------------

# Data processing

def process_face(self):
    # Process face information into more easily distinguishable dictionary format
    face = {}
    for f in range(self.num_faces):
        face[f] = {}
        for i in range(4):
            index = f * 4 + i
            face[f][i] = {}
            face[f][i]['id'] = self.face_nodes[index]
            face[f][i]['index_id'] = self.node_labels.index(self.face_nodes[index]) + 1
            face[f][i]['coord'] = self.nodes[self.node_labels.index(self.face_nodes[index])]
            face[f]['pressure'] = self.p[f]
        face[f]['norm'] = -np.cross([np.array(face[f][0]['coord'])-np.array(face[f][1]['coord'])],[np.array(face[f][0]['coord'])-np.array(face[f][3]['coord'])])[0]
        face[f]['unit_norm'] = face[f]['norm']/np.linalg.norm(face[f]['norm'])
        face[f]['px'] = face[f]['unit_norm'][0] * face[f]['pressure']
        face[f]['py'] = face[f]['unit_norm'][1] * face[f]['pressure']
        face[f]['pz'] = face[f]['unit_norm'][2] * face[f]['pressure']
        face[f]['centre'] = (face[f][0]['coord'] + face[f][1]['coord'] + face[f][2]['coord'] + face[f][3]['coord']) /4
        self.pressure_force[f,:] = [face[f]['px'],face[f]['py'],face[f]['pz']]

    face['num_faces'] = self.num_faces
    face['n_a'] = self.n_a

    return face


def calculate_pressure_force(self, p):
    for f in range(self.num_faces):
        px = self.face[f]['unit_norm'][0] * p[f]
        py = self.face[f]['unit_norm'][1] * p[f]
        pz = self.face[f]['unit_norm'][2] * p[f]

        self.pressure_force[f,:] = [px, py, pz]

    return


def get_face_centres(self):
    face_centre = np.zeros((self.num_faces, 3))
    for i in range(self.num_faces):
        face_centre[i,:] = self.face_dictionary[i]['centre']
    return face_centre


# Data writing

def write_tec(self, fname):
    f = open(fname, "w")
    f.write("TITLE = \"Surface plot\"\n")
    f.write("VARIABLES = \"X\", \"Y\", \"Z\", \"normx\", \"normy\", \"normz\", \"px\", \"py\", \"pz\"\n")
    f.write("ZONE T=\"PURE-QUADS\", NODES={}, ELEMENTS={}, DATAPACKING=BLOCK, VARLOCATION=([4,5,6,7,8,9]=CELLCENTERED), ZONETYPE=FEQUADRILATERAL\n".format(self.n_a,self.num_faces))
    for i in range(self.n_a):
        f.write("{}\n".format(self.nodes[i, 0]))
    for i in range(self.n_a):
        f.write("{}\n".format(self.nodes[i, 1]))
    for i in range(self.n_a):
        f.write("{}\n".format(self.nodes[i, 2]))
    for i in range(self.num_faces):
        f.write("{}\n".format(self.face_dictionary[i]['unit_norm'][0]))
    for i in range(self.num_faces):
        f.write("{}\n".format(self.face_dictionary[i]['unit_norm'][1]))
    for i in range(self.num_faces):
        f.write("{}\n".format(self.face_dictionary[i]['unit_norm'][2]))
    for i in range(self.num_faces):
        f.write("{}\n".format(self.pressure_force[i, 0]))
    for i in range(self.num_faces):
        f.write("{}\n".format(self.pressure_force[i, 1]))
    for i in range(self.num_faces):
        f.write("{}\n".format(self.pressure_force[i, 2]))
    for i in range(self.num_faces):
        f.write("{} {} {} {}\n".format(self.face_dictionary[i][0]['index_id'], self.face_dictionary[i][1]['index_id'], self.face_dictionary[i][2]['index_id'], self.face_dictionary[i][3]['index_id']))
    
    f.close()


def write_struc_tec(self, fname):

    # Write beam nodes to tecplot file
    f = open(fname, "w")
    f.write("TITLE = \"Beamstick model\"\n")
    f.write("VARIABLES = \"X\", \"Y\", \"Z\" \"px\", \"py\", \"pz\"\n")
    f.write("ZONE I={}, J=1, K=1\n".format(self.n_s))
    for i in range(self.n_s):
        f.write("{} {} {} {} {} {}\n".format(self.struct_nodes[i][0], self.struct_nodes[i][1], self.struct_nodes[i][2], self.F_s[i, 0], self.F_s[i, 1], self.F_s[i, 2]))
    f.close()


def write_structural_forces(self):
    f = open("pforce_struc.dat", "w")
    f.write("{}\n".format(self.num_faces))
    for i in range(self.num_faces):
        f.write("{} {} {}\n".format(self.face_dictionary[i]['px'], self.face_dictionary[i]['py'], self.face_dictionary[i]['pz']))
    f.close()


def write_surf(self):
    f = open("aero_nodes.xyz", "w")
    f.write("{}\n".format(self.n_a))
    for i in range(self.n_a):
        f.write("{} {} {}\n".format(self.nodes[i, 0], self.nodes[i, 1], self.nodes[i, 2]))
    f.close()


# Loading data


def loadbeamstick(self):
    self.struct_nodes = np.loadtxt('beamstick.dat', skiprows=4)
    self.n_s = len(self.struct_nodes[:, 0])
    beam_forward = self.struct_nodes
    beam_aft = self.struct_nodes
    for i in range(self.n_s):
        beam_forward[i, 0] = beam_forward[i, 2] + 1
        beam_aft[i, 0] = beam_aft[i, 2] - 1
    
    self.struct_nodes = np.append(self.struct_nodes, beam_forward, axis=0)
    self.struct_nodes = np.append(self.struct_nodes, beam_aft, axis=0)
    print(self.struct_nodes.shape)
    self.n_s = self.n_s * 3
    return
    
def loadDisplacements(self):
    u = np.loadtxt('deformed.dat',skiprows=4)
    u1 = u
    u = np.append(u, u1, axis=0)
    u = np.append(u, u1, axis=0)
    return u



    

uob_fsi_coupling = {'post init hook': post_init,
                       'start real time cycle hook': start_real_time_cycle,
                       'post advance hook': post_advance,
                       'post output hook': post_output}
