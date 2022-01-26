"""
Copyright (c) 2014, Zenotech Ltd
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
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

import netCDF4
import h5py
import numpy as np
from sets import Set
import sys
import os


class Cell(object):

    TETRA = 0x1
    PRISM = 0x2
    PYRA = 0x4
    HEX = 0x8
    POLY = 0x10
    NUM_TYPE = 0x5
    TYPE_MASK = 0x1F
    NUM_FACE_MASK = 0x7FF

    def get_type(cell_type):
        return cell_type & TYPE_MASK

    @classmethod
    def getNumFaces(cls, cell_type):
        return (cell_type >> NUM_TYPE) & NUM_FACE_MASK

    @classmethod
    def encode(cls, cell_type, num_faces):
        return (num_faces << 5) + cell_type


class Face(object):

    def __init__(self):
        self.left = -1
        self.right = -1
        self.zone = -1
        self.nodes = [-1, -1, -1, -1]

    def to_array(self, face_array):
        face_array[0] = self.left
        face_array[1] = self.right
        face_array[2] = self.zone
        for i in range(3, 7):
            face_array[i] = self.nodes[i - 3]

    def from_array(self, face_array):
        self.left = face_array[0]
        self.right = face_array[1]
        self.zone = face_array[2]
        for i in range(3, 7):
            self.nodes[i - 3] = face_array[i]

    def get_node(self, face_array, index):
        return face_array[index + 3]


class ProgressBar(object):

    def __init__(self):
        self.percent = 0

    def reset(self):
        self.percent = 0

    def update(self, percent, bar_length=40):
        new_percent = int(round(percent * 100))
        if new_percent > self.percent:
            self.percent = new_percent
            hashes = '#' * int(round(percent * bar_length))
            spaces = ' ' * (bar_length - len(hashes))
            sys.stdout.write("\rPercent: [{0}] {1}%".format(
                hashes + spaces, int(round(percent * 100))))
            sys.stdout.flush()


class TauTozCFD(object):

    def reorder(self, nodes):

        reordered = [-1, -1, -1, -1]

        num_nodes = 4

        if nodes[3] == -1:
            num_nodes = 3

        min_index = 0
        min_val = nodes[0]

        for i in range(1, num_nodes):
            if nodes[i] < min_val:
                min_index = i
                min_val = nodes[i]

        assert min_val >= 0

        j = 0
        for i in range(min_index, num_nodes):
            reordered[j] = nodes[i]
            j += 1

        for i in range(0, min_index):
            reordered[j] = nodes[i]
            j += 1

        for i in range(0, num_nodes):
            nodes[i] = reordered[i]

    def write_points(self):
        # Write points
        print('Writing Points')

        dset = self.grp.create_dataset(
            "nodeVertex", (len(self.num_points), 3), dtype='f8')

        coords = np.empty((len(self.num_points), 3), dtype='f8')

        points_xc = self.nc.variables['points_xc'][:]
        points_yc = self.nc.variables['points_yc'][:]
        points_zc = self.nc.variables['points_zc'][:]

        for i in range(0, len(self.num_points)):
            coords[i][0] = points_xc[i]
            coords[i][1] = points_yc[i]
            coords[i][2] = points_zc[i]

        dset[...] = coords

        print('Completed writing points')

    def update_face_array(self, face):
        face.to_array(self.face_array[self.face_count])
        self.face_count += 1

    def surface_triangles(self):

        print('Surface triangles')

        points_of_surfacetriangles = self.nc.variables['points_of_surfacetriangles'][:]

        for i in range(0, len(self.num_surface_tri)):
            f = Face()
            N1 = points_of_surfacetriangles[i][0]
            N2 = points_of_surfacetriangles[i][1]
            N3 = points_of_surfacetriangles[i][2]
            # f.nodes[0] = N1
            # f.nodes[1] = N2
            # f.nodes[2] = N3
            f.nodes[0] = N3
            f.nodes[1] = N2
            f.nodes[2] = N1

            # print f.nodes
            self.reorder(f.nodes)
            # print f.nodes

            # face_list.append(f)

            self.update_face_array(f)

            self.progress.update(float(i) / len(self.num_surface_tri))

    def surface_quads(self):

        print('Surface quads')

        points_of_surfacequadrilaterals = self.nc.variables['points_of_surfacequadrilaterals'][:]

        for i in range(0, len(self.num_surface_quad)):
            f = Face()
            N1 = points_of_surfacequadrilaterals[i][0]
            N2 = points_of_surfacequadrilaterals[i][1]
            N3 = points_of_surfacequadrilaterals[i][2]
            N4 = points_of_surfacequadrilaterals[i][3]
            # f.nodes[0] = N1
            # f.nodes[1] = N2
            # f.nodes[2] = N3
            # f.nodes[3] = N4
            f.nodes[0] = N4
            f.nodes[1] = N3
            f.nodes[2] = N2
            f.nodes[3] = N1

            # print f.nodes
            self.reorder(f.nodes)
            # print f.nodes

            # face_list.append(f)

            self.update_face_array(f)

            self.progress.update(float(i) / len(self.num_surface_quad))

    def tetrahedra(self):

        print('Tetrahedra')

        """
         * Tetrahedron
         *
         *          3
         *         /|\
         *        / | \
         *       /  |  \
         *      0---|---2
         *        \ | /
         *          1
         *
         """
        points_of_tets = self.nc.variables['points_of_tetraeders'][:]

        f = Face()

        for i in range(0, len(self.num_tets)):
            N1 = points_of_tets[i][0]
            N2 = points_of_tets[i][1]
            N3 = points_of_tets[i][2]
            N4 = points_of_tets[i][3]
            # N1,N3,N2
            f.left = self.num_cells
            f.nodes[0] = N1
            f.nodes[1] = N3
            f.nodes[2] = N2
            f.nodes[3] = -1
            self.reorder(f.nodes)
            # face_list.append(f)
            self.update_face_array(f)

            # N1,N2,N4
            # f = Face()
            f.left = self.num_cells
            f.nodes[0] = N1
            f.nodes[1] = N2
            f.nodes[2] = N4
            f.nodes[3] = -1
            self.reorder(f.nodes)
            # face_list.append(f)
            self.update_face_array(f)
            # N2,N3,N4
            # f = Face()
            f.left = self.num_cells
            f.nodes[0] = N2
            f.nodes[1] = N3
            f.nodes[2] = N4
            f.nodes[3] = -1
            self.reorder(f.nodes)
            # face_list.append(f)
            self.update_face_array(f)
            # N3,N1,N4
            # f = Face()
            f.left = self.num_cells
            f.nodes[0] = N3
            f.nodes[1] = N1
            f.nodes[2] = N4
            f.nodes[3] = -1
            self.reorder(f.nodes)
            # face_list.append(f)
            self.update_face_array(f)

            self.num_cells += 1

            self.progress.update(float(i) / len(self.num_tets))

    def prism(self):

        print('Prisms')
        """
         * Prism                  face schematic
         *                               0
         *     3-------5          0-------------2
         *     |\     /|           \     4     /
         *     | \  /  |             3-------5
         *     |  4    |              \  1  /
         *     |  |    |            2  \  /   3
         *     0--|----2                4
         *      \ |   /                 |
         *       \| /                   1
         *        1
         *
         """
        points_of_prisms = self.nc.variables['points_of_prisms'][:]

        f = Face()
        for i in range(0, len(self.num_prism)):
            N1 = points_of_prisms[i][0]
            N2 = points_of_prisms[i][1]
            N3 = points_of_prisms[i][2]
            N4 = points_of_prisms[i][3]
            N5 = points_of_prisms[i][4]
            N6 = points_of_prisms[i][5]
            # N1,N3,N2
            f.left = self.num_cells
            f.nodes[0] = N1
            f.nodes[1] = N3
            f.nodes[2] = N2
            f.nodes[3] = -1
            self.reorder(f.nodes)
            # face_list.append(f)
            self.update_face_array(f)
            # N4,N5,N6
            # f = Face()
            f.left = self.num_cells
            f.nodes[0] = N4
            f.nodes[1] = N5
            f.nodes[2] = N6
            f.nodes[3] = -1
            self.reorder(f.nodes)
            # face_list.append(f)
            self.update_face_array(f)
            # N1,N2,N5,N4
            # f = Face()
            f.left = self.num_cells
            f.nodes[0] = N1
            f.nodes[1] = N2
            f.nodes[2] = N5
            f.nodes[3] = N4
            self.reorder(f.nodes)
            # face_list.append(f)
            self.update_face_array(f)
            # N2,N3,N6,N5
            # f = Face()
            f.left = self.num_cells
            f.nodes[0] = N2
            f.nodes[1] = N3
            f.nodes[2] = N6
            f.nodes[3] = N5
            self.reorder(f.nodes)
            # face_list.append(f)
            self.update_face_array(f)
            # N3,N1,N4,N6
            # f = Face()
            f.left = self.num_cells
            f.nodes[0] = N3
            f.nodes[1] = N1
            f.nodes[2] = N4
            f.nodes[3] = N6
            self.reorder(f.nodes)
            # face_list.append(f)
            self.update_face_array(f)

            self.num_cells += 1

            self.progress.update(float(i) / len(self.num_prism))

    def pyramids(self):

        print('Pyramids')
        """
         * Pyramid                   faces
         *            4
         *          /|            0-------3
         *         / |            |\     /|
         *        / |.|           | \ 3 / |
         *       /  |.|           |  \ /  |
         *      /  |  .|          | 4 4 2 | 0
         *     /   |  .|          |  / \  |
         *    0---|---3 |         | /   \ |
         *     \  |    \|         |/  1  \|
         *      \|      \|        1-------2
         *       1-------2
         *
         """

        points_of_pyramids = self.nc.variables['points_of_pyramids'][:]

        f = Face()
        for i in range(0, len(self.num_pyramid)):
            N1 = points_of_pyramids[i][0]
            N2 = points_of_pyramids[i][1]
            N3 = points_of_pyramids[i][2]
            N4 = points_of_pyramids[i][3]
            N5 = points_of_pyramids[i][4]
            # N1,N5,N4
            f.left = self.num_cells
            f.nodes[0] = N1
            f.nodes[1] = N5
            f.nodes[2] = N4
            f.nodes[3] = -1
            self.reorder(f.nodes)
            # face_list.append(f)
            self.update_face_array(f)
            # N1,N2,N5
            # f = Face()
            f.left = self.num_cells
            f.nodes[0] = N1
            f.nodes[1] = N2
            f.nodes[2] = N5
            f.nodes[3] = -1
            self.reorder(f.nodes)
            # face_list.append(f)
            self.update_face_array(f)
            # N2,N3,N5
            # f = Face()
            f.left = self.num_cells
            f.nodes[0] = N2
            f.nodes[1] = N3
            f.nodes[2] = N5
            f.nodes[3] = -1
            self.reorder(f.nodes)
            # face_list.append(f)
            self.update_face_array(f)
            # N3,N4,N5
            # f = Face()
            f.left = self.num_cells
            f.nodes[0] = N3
            f.nodes[1] = N4
            f.nodes[2] = N5
            f.nodes[3] = -1
            self.reorder(f.nodes)
            # face_list.append(f)
            self.update_face_array(f)
            # N1,N4,N3,N2
            # f = Face()
            f.left = self.num_cells
            f.nodes[0] = N1
            f.nodes[1] = N4
            f.nodes[2] = N3
            f.nodes[3] = N2
            self.reorder(f.nodes)
            # face_list.append(f)
            self.update_face_array(f)

            self.num_cells += 1

            self.progress.update(float(i) / len(self.num_pyramid))

    def hexahedra(self):

        print('Hexahedra')
        """
         * Hexahedron
         *                             faces:
         *    4-------7         0----------------3
         *    |\      |\        |\              /|
         *    | \     | \       | \      3     / |
         *    |  5-------6      |  \4--------7/  |
         *    |  |    |  |      |   |        |   |
         *    0--|----3  |      | 5 |   4    | 2 | 0
         *     \ |     \ |      |   |        |   |
         *      \|      \|      |  /5--------6\  |
         *       1-------2      | /     1      \ |
         *                      |/              \|
         *                      1----------------2
         *
        """
        points_of_hexaeders = self.nc.variables['points_of_hexaeders'][:]

        f = Face()
        for i in range(0, len(self.num_hex)):
            N1 = points_of_hexaeders[i][0]
            N2 = points_of_hexaeders[i][1]
            N3 = points_of_hexaeders[i][2]
            N4 = points_of_hexaeders[i][3]
            N5 = points_of_hexaeders[i][4]
            N6 = points_of_hexaeders[i][5]
            N7 = points_of_hexaeders[i][6]
            N8 = points_of_hexaeders[i][7]

            # N1,N4,N3,N2
            f.left = self.num_cells
            f.nodes[0] = N1
            f.nodes[1] = N4
            f.nodes[2] = N3
            f.nodes[3] = N2
            self.reorder(f.nodes)
            # face_list.append(f)
            self.update_face_array(f)
            # N1,N2,N6,N5
            # f = Face()
            f.left = self.num_cells
            f.nodes[0] = N1
            f.nodes[1] = N2
            f.nodes[2] = N6
            f.nodes[3] = N5
            self.reorder(f.nodes)
            # face_list.append(f)
            self.update_face_array(f)
            # N2,N3,N7,N6
            # f = Face()
            f.left = self.num_cells
            f.nodes[0] = N2
            f.nodes[1] = N3
            f.nodes[2] = N7
            f.nodes[3] = N6
            self.reorder(f.nodes)
            # face_list.append(f)
            self.update_face_array(f)
            # N3,N4,N8,N7
            # f = Face()
            f.left = self.num_cells
            f.nodes[0] = N3
            f.nodes[1] = N4
            f.nodes[2] = N8
            f.nodes[3] = N7
            self.reorder(f.nodes)
            # face_list.append(f)
            self.update_face_array(f)
            # N1,N5,N8,N4
            # f = Face()
            f.left = self.num_cells
            f.nodes[0] = N1
            f.nodes[1] = N5
            f.nodes[2] = N8
            f.nodes[3] = N4
            self.reorder(f.nodes)
            # face_list.append(f)
            self.update_face_array(f)
            # N5,N6,N7,N8
            # f = Face()
            f.left = self.num_cells
            f.nodes[0] = N5
            f.nodes[1] = N6
            f.nodes[2] = N7
            f.nodes[3] = N8
            self.reorder(f.nodes)
            # face_list.append(f)
            self.update_face_array(f)

            self.num_cells += 1

            self.progress.update(float(i) / len(self.num_hex))

    def main(self, tau_mesh, zcfd_mesh):

        self.nc = netCDF4.Dataset(tau_mesh, 'r')

        self.num_elements = self.nc.dimensions['no_of_elements']
        self.num_tets = self.nc.dimensions['no_of_tetraeders']
        self.num_prism = self.nc.dimensions['no_of_prisms']
        self.num_pyramid = self.nc.dimensions['no_of_pyramids']
        self.num_hex = self.nc.dimensions['no_of_hexaeders']

        self.num_surface_elements = self.nc.dimensions['no_of_surfaceelements']
        self.num_surface_tri = self.nc.dimensions['no_of_surfacetriangles']
        self.num_surface_quad = self.nc.dimensions['no_of_surfacequadrilaterals']
        self.num_points = self.nc.dimensions['no_of_points']

        self.face_array = np.empty((len(self.num_surface_tri) +
                                    len(self.num_surface_quad) +
                                    4 * len(self.num_tets) +
                                    5 * len(self.num_prism) +
                                    5 * len(self.num_pyramid) +
                                    6 * len(self.num_hex), 7), dtype='i')

        self.progress = ProgressBar()

        print('Extracting faces')

        self.face_count = 0
        self.num_cells = 0

        self.surface_triangles()
        print(" ")
        self.progress.reset()
        self.surface_quads()
        print(" ")
        self.progress.reset()

        print('Boundary marker')

        boundarymarker_of_surfaces = self.nc.variables['boundarymarker_of_surfaces'][:]

        self.zones = Set()
        for i in range(0, len(self.num_surface_elements)):
            z = boundarymarker_of_surfaces[i]
            self.face_array[i][2] = z
            self.zones.add(z)

        print('Number of unique zones ') + str(len(self.zones))
        self.tetrahedra()
        print(" ")
        self.progress.reset()
        self.prism()
        print(" ")
        self.progress.reset()
        self.pyramids()
        print(" ")
        self.progress.reset()
        self.hexahedra()
        print(" ")
        self.progress.reset()

        num_faces = len(self.face_array) / 2

        print('Number of faces: ') + str(num_faces)

        print('Building node to face pointer')
        node_to_face = []
        for i in range(0, len(self.num_points)):
            node_to_face.append([])

        f = Face()
        for i in range(0, len(self.face_array)):
            n = f.get_node(self.face_array[i], 0)
            node_to_face[n].append(i)

        print('Matching faces')

        f_list = []
        bf_list = []

        while self.face_count > 0:
            found = False

            f = Face()

            self.face_count -= 1
            f.from_array(self.face_array[self.face_count])
            # pop face

            n = f.nodes[0]

            if n != -1:
                for i in range(0, len(node_to_face[n])):
                    nn = node_to_face[n][i]

                    if nn != -1 and nn < self.face_count:
                        # Potential neighbour face
                        fn = Face()
                        fn.from_array(self.face_array[nn])

                        if f.nodes[3] != -1:  # Quad
                            if f.nodes[0] == fn.nodes[0]:
                                if f.nodes[1] == fn.nodes[3] and f.nodes[2] == fn.nodes[2] and f.nodes[3] == fn.nodes[1]:

                                    found = True
                                    f.right = fn.left
                                    f.zone = fn.zone
                                    node_to_face[n][i] = -1
                                    fn.nodes[0] = -1
                                    fn.to_array(self.face_array[nn])
                                    break

                        elif f.nodes[3] == -1 and fn.nodes[3] == -1:  # Triangle
                            if f.nodes[0] == fn.nodes[0]:
                                if f.nodes[1] == fn.nodes[2] and f.nodes[2] == fn.nodes[1]:

                                    found = True
                                    f.right = fn.left
                                    f.zone = fn.zone
                                    node_to_face[n][i] = -1
                                    fn.nodes[0] = -1
                                    fn.to_array(self.face_array[nn])
                                    break

                if not found:
                    print(f.nodes)
                    # printf("%d %d %d %d %d\n",f.nodes[0],f.nodes[1],f.nodes[2],f.nodes[3],facelst.size());
                    for i in range(0, len(node_to_face[n])):
                        nn = node_to_face[n][i]
                        # print face_list[nn].nodes
                        # printf("%d %d %d %d %d\n",facelst[nn].nodes[0],facelst[nn].nodes[1],facelst[nn].nodes[2],facelst[nn].nodes[3],facelst[nn].zone);

                assert found

                if f.zone == -1:
                    assert f.right != -1
                    f_list.append(f)
                else:
                    assert f.right == -1
                    bf_list.append(f)

                # if len(f_list)%100000 == 0:
                #    print len(f_list)

                self.progress.update(
                    float(num_faces - self.face_count / 2) / num_faces)

        print(" ")
        self.progress.reset()

        # Add boundary faces
        for i in range(0, len(bf_list)):
            f_list.append(bf_list[i])

        # Check we have extracted correct number of faces
        assert len(f_list) == num_faces

        # Extra checks and set halo cells
        for i in range(0, len(f_list)):

            assert f_list[i].left != -1
            if f_list[i].zone == -1:
                assert f_list[i].right != -1
            if f_list[i].right == -1:
                f_list[i].right = self.num_cells
                self.num_cells += 1

        # Check number of cells
        assert self.num_cells == len(
            self.num_elements) + len(self.num_surface_elements)

        num_faces = len(f_list)
        num_cells = len(self.num_elements)

        # Write zCFD mesh
        print('Writing zCFD mesh')

        f = h5py.File(zcfd_mesh, "w")

        self.grp = f.create_group("mesh")

        self.write_points()

        print('Writing faces')

        self.grp.attrs['numFaces'] = num_faces
        self.grp.attrs['numCells'] = num_cells

        face_to_cell = np.empty((num_faces, 2), dtype='i')

        for i in range(0, num_faces):
            face_to_cell[i][0] = f_list[i].left
            face_to_cell[i][1] = f_list[i].right

        dset = self.grp.create_dataset("faceCell", (num_faces, 2), dtype='i')
        dset[...] = face_to_cell

        face_type = np.full((num_faces), 4, dtype='i')
        for i in range(0, num_faces):
            if f_list[i].nodes[3] == -1:
                face_type[i] = 3

        dset = self.grp.create_dataset("faceType", (num_faces,), dtype='i')
        dset[...] = face_type

        face_nodes_list = []  # np.empty((4*num_faces),dtype='i')
        for i in range(0, num_faces):
            if f_list[i].nodes[3] != -1:
                for j in range(0, 4):
                    face_nodes_list.append(f_list[i].nodes[j])
            else:
                for j in range(0, 3):
                    face_nodes_list.append(f_list[i].nodes[j])

            self.progress.update(float(i) / num_faces)
        print(" ")
        self.progress.reset()

        face_nodes = np.empty((len(face_nodes_list)), dtype='i')
        for i in range(0, len(face_nodes_list)):
            face_nodes[i] = face_nodes_list[i]

        dset = self.grp.create_dataset(
            "faceNodes", (len(face_nodes_list),), dtype='i')
        dset[...] = face_nodes

        face_bc = np.full((num_faces), 3, dtype='i')
        for i in range(0, num_faces):
            if f_list[i].zone == -1:
                face_bc[i] = 0

        # Attempt to read bmap file to get boundary conditions
        name = os.path.split(tau_mesh)[1]
        basename = name.split('.')[0]
        bmap_filename = os.path.join(
            os.path.split(tau_mesh)[0], basename + ".bmap")
        if os.path.isfile(bmap_filename):
            bmap_file = open(bmap_filename)

        dset = self.grp.create_dataset("faceBC", (num_faces,), dtype='i')
        dset[...] = face_bc

        face_info = np.empty((num_faces, 2), dtype='i')
        for i in range(0, num_faces):
            z = f_list[i].zone
            if z == -1:
                for z in range(0, len(self.zones) + 1):
                    if z not in self.zones:
                        break

            face_info[i][0] = z
            face_info[i][1] = 0

        dset = self.grp.create_dataset("faceInfo", (num_faces, 2), dtype='i')
        dset[...] = face_info

        cell_type = np.empty((len(self.num_elements)), dtype='i')
        count = 0
        for i in range(0, len(self.num_tets)):
            cell_type[count] = Cell.encode(Cell.TETRA, 4)
            count += 1
        for i in range(0, len(self.num_prism)):
            cell_type[count] = Cell.encode(Cell.PRISM, 5)
            count += 1
        for i in range(0, len(self.num_pyramid)):
            cell_type[count] = Cell.encode(Cell.PYRA, 5)
            count += 1
        for i in range(0, len(self.num_hex)):
            cell_type[count] = Cell.encode(Cell.HEX, 6)
            count += 1

        dset = self.grp.create_dataset(
            "cellType", (len(self.num_elements),), dtype='i')
        dset[...] = cell_type

        print('Conversion Complete')


if __name__ == "__main__":

    print('DLR Tau to zCFD mesh converter')

    if len(sys.argv) != 3:
        print('Usage: TautozCFD tau_mesh_name zcfd_mesh_name')
    else:
        converter = TauTozCFD()
        converter.main(sys.argv[1], sys.argv[2])
