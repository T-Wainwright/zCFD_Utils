import numpy as np
import pyNastran
from pyNastran.op2.op2 import read_op2
from pyNastran.bdf.bdf import BDF
import os
from zcfd.utils import config
from scipy.io import loadmat


# class to read cba modal results file


class cba_modal:
    def __init__(self, fname=None) -> None:
        if fname:
            self.read_modes(fname)
        else:
            pass

    def read_modes(self, fname):
        row_ctr = 2
        self.num_modes = int(np.loadtxt(fname, skiprows=row_ctr, max_rows=1))
        row_ctr += 2

        self.eigenvalues = np.loadtxt(fname, skiprows=row_ctr, max_rows=self.num_modes)

        row_ctr += self.num_modes + 1

        self.num_grid_points = int(np.loadtxt(fname, skiprows=row_ctr, max_rows=1))

        row_ctr += 2

        self.num_Dof = int(np.loadtxt(fname, skiprows=row_ctr, max_rows=1))

        row_ctr += 2

        self.Dof = np.loadtxt(fname, skiprows=row_ctr, max_rows=1)

        row_ctr += 3

        self.mode_data = np.zeros((self.num_modes, self.num_grid_points, self.num_Dof))

        # read eigen vectors
        for i in range(self.num_modes):
            self.mode_data[i, :, :] = np.loadtxt(
                fname, skiprows=row_ctr, max_rows=self.num_grid_points
            )
            row_ctr += self.num_grid_points + 1

        self.grid_points = np.loadtxt(
            fname, skiprows=row_ctr, max_rows=self.num_grid_points
        )

        self.calculate_mode_frequencies()

        self.M = np.identity(self.num_modes)
        self.C = np.diag([0 for i in range(self.num_modes)])
        self.K = np.diag([(l / (2 * np.pi)) ** 2 for l in self.mode_freqs])

    def calculate_norms(self):
        self.norms = np.zeros((self.num_modes, self.num_grid_points))
        for i in range(self.num_modes):
            for j in range(self.num_grid_points):
                self.norms[i, j] = np.linalg.norm(self.mode_data[i, j, :])

    def write_grid_tec(self, fname):
        f = open(fname, "w")
        for i in range(self.num_grid_points):
            f.write(
                "{}, {}, {}\n".format(
                    self.grid_points[i, 0],
                    self.grid_points[i, 1],
                    self.grid_points[i, 2],
                )
            )
        f.close()

    def write_grid_csv(self, fname: str):
        f = open(fname, "w")
        f.write("X, Y, Z, ")
        for i in range(self.num_modes):
            f.write("m{}X, m{}Y, m{}Z, ".format(i, i, i))
        f.write("\n")
        # dump points file
        for i in range(self.num_grid_points):
            f.write(
                "{}, {}, {}, ".format(
                    self.grid_points[i, 0],
                    self.grid_points[i, 1],
                    self.grid_points[i, 2],
                )
            )
            for j in range(self.num_modes):
                for k in range(3):
                    f.write("{}, ".format(self.mode_data[j, i, k]))
            f.write("\n")

        f.close()

    def calculate_mode_frequencies(self):
        self.mode_freqs = np.array([np.sqrt(i) for i in self.eigenvalues])

    def write_modes(self, fname="modes.cba"):
        with open(fname, "w") as f:
            f.write("1\n")
            f.write("Number of Modes\n")
            f.write("{}\n".format(self.num_modes))
            f.write("eigenvalues:\n")
            for e in self.eigenvalues:
                f.write("{}\n".format(e))
            f.write("PLT1\n")
            f.write("{}\n".format(self.num_grid_points))
            f.write("Numer of Dof\n")
            f.write("{}\n".format(self.num_modes))
            f.write("degrees of freedom (x=1, y=2, z=3, rotx=4, roty=5, rotz=6)\n")
            for d in self.Dof:
                f.write("{} ".format(int(d)))
            f.write("\n")
            f.write("eigenvectors: \n")
            for m in range(self.num_modes):
                f.write("{}\n".format(m))
                for i in range(self.num_grid_points):
                    for j in range(self.num_modes):
                        f.write("{} \t".format(self.mode_data[m, i, j]))
                    f.write("\n")
            f.write("grid\n")
            for i in range(self.num_grid_points):
                for j in range(3):
                    f.write("{} \t".format(self.grid_points[i, j]))
                f.write("\n")

    def get_moving_nodes(self):
        return self.grid_points

    def get_loading_nodes(self):
        return self.grid_points


class NastranReader:
    def __init__(self, filename):
        self.filename = filename
        print(" Reading Nastran data from {}".format(self.filename))
        self.op2 = read_op2(filename + ".op2", build_dataframe=False, debug=False)
        self.bdf = BDF(debug=False)

    def read_grid_points(self):
        print(" Reading grid points from %s" % self.filename + ".bdf")
        self.bdf.read_bdf(self.filename + ".bdf")

        count = 0
        self.structural_nodes = set()

        mass_elements = ["CONM1", "CONM2", "CMASS1", "CMASS2", "CMASS3", "CMASS4"]

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

        print(" Read %i grid points" % self.num_grid_points)

        self.grid_points = grid_points

    def read_modes(self, mode_list=None, modal_damping=None):
        num_cases = len(list(self.op2.eigenvectors.keys()))
        print(" Reading %i loadcase(s) from %s" % (num_cases, self.filename + ".op2"))

        if num_cases > 1:
            print(" More than one load case in %s" % self.filename)

        eig1 = self.op2.eigenvectors[1]
        self.mode_freqs = list(eig1.mode_cycles)
        for i in range(len(self.mode_freqs)):
            self.mode_freqs[i] *= 2.0 * np.pi
        self.eigenvals = eig1.eigns
        self.num_modes = len(eig1.modes)

        # Mode selection
        if not mode_list:
            mode_list = [ii for ii in range(self.num_modes)]
        else:
            self.num_modes = len(mode_list)

        node_id_to_index = {}
        for i, node_id in enumerate(self.bdf.nodes.keys()):
            node_id_to_index[node_id] = i

        mode_data = np.zeros((self.num_modes, self.num_grid_points, 6))
        for ii in range(self.num_modes):
            for jj, node_id in enumerate(self.structural_nodes):
                op2_index = node_id_to_index[node_id]
                mode_data[ii, jj, :] = eig1.data[mode_list[ii], op2_index, :]

        self.mode_freqs = [self.mode_freqs[i] for i in mode_list]

        # Modal damping
        if not modal_damping:
            self.mode_damping = [0.0] * self.num_modes
        else:
            self.mode_damping = modal_damping

        print(" Read %i modes" % self.num_modes)
        print(
            " Modal frequencies (Hz) %s"
            % " ".join(
                ["{: .2f}".format(freq / (2.0 * np.pi)) for freq in self.mode_freqs]
            )
        )

        self.mode_data = mode_data

        self.M = np.identity(self.num_modes)
        self.C = np.diag(self.mode_damping)
        self.K = np.diag([(l / (2 * np.pi)) ** 2 for l in self.mode_freqs])

    def print_op2_stats(self, short=False):
        print(self.op2.get_op2_stats(short), flush=True)

    def print_bdf(self):
        print(self.bdf.object_attributes(), flush=True)
        print(self.bdf.object_methods(), flush=True)

    def get_moving_nodes(self):
        return self.grid_points

    def get_loading_nodes(self):
        return self.grid_points


class atomReader:
    def __init__(self, fname, pitch=0.0) -> None:
        ATOM_FE = loadmat(fname, squeeze_me=False)
        ATOM_Modal = ATOM_FE["Modal"]
        # Got to have some gross indexing in here due to pythons loadmat behavior
        atom_grid_points = ATOM_FE["BeamAxis_123"]
        self.num_modes = int(ATOM_Modal["nEigval"][0][0][0][0])
        self.num_grid_points = atom_grid_points.shape[0]

        self.M = ATOM_Modal["Mmodal"][0][0]
        self.C = ATOM_Modal["Cmodal"][0][0]
        self.K = ATOM_Modal["Kmodal"][0][0]

        self.mode_data_mat = ATOM_Modal["Eigvec"][0][0]

        mode_data_atom = np.zeros((self.num_modes, self.num_grid_points, 6))

        for m in range(self.num_modes):
            for i in range(self.num_grid_points - 1):
                for j in range(6):
                    mode_data_atom[m, i + 1, j] = self.mode_data_mat[6 * i + j, m]

        mode_freq = []
        for i in range(self.num_modes):
            mode_freq.append(ATOM_Modal["ModeShape"][0][0][i][0][0][0][0])

        self.mode_freqs = np.array(mode_freq)

        # ATOM uses a blade centric coordinate system:
        # x positive along blade
        # y positive in flow direction
        # z positive in a right hand set

        # CFD uses a tower centric coorcinate system:
        # x positive in flow direction
        # y positive in a right hand set
        # z positive along the blade

        self.grid_points = np.zeros((self.num_grid_points, 3))
        self.grid_points[:, 0] = atom_grid_points[:, 1]
        self.grid_points[:, 1] = atom_grid_points[:, 2]
        self.grid_points[:, 2] = atom_grid_points[:, 0] + 3.0

        # rotate grid points about pitch axis

        self.mode_data = np.zeros((self.num_modes, self.num_grid_points, 6))
        self.mode_data[:, :, 0] = mode_data_atom[:, :, 1]
        self.mode_data[:, :, 1] = mode_data_atom[:, :, 2]
        self.mode_data[:, :, 2] = mode_data_atom[:, :, 0]

        self.mode_data[:, :, 3] = mode_data_atom[:, :, 4]
        self.mode_data[:, :, 4] = mode_data_atom[:, :, 5]
        self.mode_data[:, :, 5] = mode_data_atom[:, :, 3]

        self.add_ribs()

        # rotate model about pitch axis
        self.grid_points = self.grid_points @ self.R_z(np.deg2rad(pitch)).T
        self.mode_data[:, :, 0:3] = self.mode_data[:, :, 0:3] @ self.R_z(
            np.deg2rad(pitch).T
        )
        self.mode_data[:, :, 3:6] = self.mode_data[:, :, 3:6] @ self.R_z(
            np.deg2rad(pitch).T
        )
        self.rib_nodes = self.rib_nodes @ self.R_z(np.deg2rad(pitch)).T

    def add_ribs(self, rib_length=1):
        self.rib_nodes = np.zeros((self.num_grid_points * 4, 3))
        N_offset = [1 * rib_length, 0, 0]
        S_offset = [-1 * rib_length, 0, 0]
        E_offset = [0, 2 * rib_length, 0]
        W_offset = [0, -1 * rib_length, 0]

        for i in range(self.num_grid_points):
            self.rib_nodes[i * 4 + 0, :] = self.grid_points[i, :] + N_offset
            self.rib_nodes[i * 4 + 1, :] = self.grid_points[i, :] + S_offset
            self.rib_nodes[i * 4 + 2, :] = self.grid_points[i, :] + E_offset
            self.rib_nodes[i * 4 + 3, :] = self.grid_points[i, :] + W_offset

    def write_grid_vtk(self, fname="modal_model_grid.vtk"):
        nodes = self.grid_points
        elements = [[i, i + 1] for i in range(self.num_grid_points - 1)]
        print(elements)

        # add rib elements
        nodes = np.concatenate((nodes, self.rib_nodes), axis=0)
        for i in range(self.num_grid_points):
            for j in range(4):
                elements.append([i, self.num_grid_points + (i * 4) + j])

        cell_types = np.array([3 for i in range(len(elements))])

        write_vtk_fe_mesh(nodes, elements, cell_types, fname)

    def write_grid_deformed_vtk(self, displacements, froot="modal_model_deformed"):
        # deform main nodes
        nodes = self.grid_points + displacements[:, 0:3]
        elements = [[i, i + 1] for i in range(self.num_grid_points - 1)]
        print(elements)

        # deform rib nodes
        rib_displacements, rib_displacements_twist = self.deform_ribs(displacements)
        nodes_full = np.concatenate(
            (nodes, self.rib_nodes + rib_displacements + rib_displacements_twist),
            axis=0,
        )
        nodes_trans = np.concatenate(
            (nodes, self.rib_nodes + rib_displacements), axis=0
        )
        nodes_rot = np.concatenate(
            (self.grid_points, self.rib_nodes + rib_displacements_twist), axis=0
        )
        for i in range(self.num_grid_points):
            for j in range(4):
                elements.append([i, self.num_grid_points + (i * 4) + j])

        cell_types = np.array([3 for i in range(len(elements))])

        write_vtk_fe_mesh(nodes_full, elements, cell_types, froot + "_full.vtk")
        write_vtk_fe_mesh(nodes_trans, elements, cell_types, froot + "_trans.vtk")
        write_vtk_fe_mesh(nodes_rot, elements, cell_types, froot + "_rot.vtk")

    def write_mode_vtk(self, scale_factor=1.0):
        if not os.path.exists("mode_shapes_22/"):
            os.makedirs("mode_shapes_22/")

        for m in range(self.num_modes):
            displacements = scale_factor * self.mode_data[m, :, :]
            self.write_grid_deformed_vtk(
                displacements, froot="mode_shapes_22/mode_{}".format(m)
            )

    def deform_ribs(self, displacements):
        rib_displacements = np.zeros((self.num_grid_points * 4, 3))
        # pure displacements
        for i in range(self.num_grid_points):
            rib_displacements[i * 4 + 0, :] = displacements[i, :3]
            rib_displacements[i * 4 + 1, :] = displacements[i, :3]
            rib_displacements[i * 4 + 2, :] = displacements[i, :3]
            rib_displacements[i * 4 + 3, :] = displacements[i, :3]

        rib_displacements_twist = np.zeros_like(rib_displacements)
        # pure twist
        for i in range(self.num_grid_points):
            for j in range(4):
                # since we rotate about centre node, need to first translate system to centre around (0, 0)
                rib_origin = self.rib_nodes[i * 4 + j, :] - self.grid_points[i, :]
                # next rotate system about origin
                rib_origin_rot = self.R_z(displacements[i, 5]) @ rib_origin
                # move system back to original location
                rib_rot = rib_origin_rot + self.grid_points[i, :]
                # get delta between the two nodes
                rib_rot_delta = rib_rot - self.rib_nodes[i * 4 + j, :]
                # store result
                rib_displacements_twist[i * 4 + j, :] = rib_rot_delta

        return rib_displacements, rib_displacements_twist

    def R_z(self, a):
        r = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        return r

    def get_moving_nodes(self):
        return np.concatenate((self.grid_points, self.rib_nodes), axis=0)

    def get_loading_nodes(self):
        return self.grid_points


def write_vtk_fe_mesh(nodes, elements, cell_types, file_path):
    with open(file_path, "w") as vtk_file:
        vtk_file.write("# vtk DataFile Version 4.2\n")
        vtk_file.write("FE Mesh Data\n")
        vtk_file.write("ASCII\n")
        vtk_file.write("DATASET UNSTRUCTURED_GRID\n")

        # Write the node coordinates
        vtk_file.write(f"POINTS {len(nodes)} double\n")
        for node in nodes:
            vtk_file.write(f"{node[0]} {node[1]} {node[2]}\n")

        # Write the element connectivity
        num_elements = len(elements)
        total_num_entries = sum(
            len(cell) + 1 for cell in elements
        )  # Sum of nodes + 1 for the cell type
        vtk_file.write(f"\nCELLS {num_elements} {total_num_entries}\n")
        for element in elements:
            num_nodes = len(element)
            vtk_file.write(f"{num_nodes} {' '.join(str(node) for node in element)}\n")

        # Write the cell types
        vtk_file.write(f"\nCELL_TYPES {num_elements}\n")
        for cell_type in cell_types:
            vtk_file.write(f"{cell_type}\n")

    print(f"VTK FE mesh has been saved to {file_path}")
