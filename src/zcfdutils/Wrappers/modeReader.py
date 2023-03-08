import numpy as np
import pyNastran
from pyNastran.op2.op2 import read_op2
from pyNastran.bdf.bdf import BDF
import os
from zcfd.utils import config



# class to read cba modal results file


class cba_modal():
    def __init__(self, fname=None) -> None:
        if fname:
            self.read_modes(fname)
        else:
            pass

    def read_modes(self, fname):
        row_ctr = 2
        self.num_modes = int(np.loadtxt(fname, skiprows=row_ctr, max_rows=1))
        row_ctr += 2

        self.eigenvalues = np.loadtxt(
            fname, skiprows=row_ctr, max_rows=self.num_modes)

        row_ctr += self.num_modes + 1

        self.num_grid_points = int(np.loadtxt(
            fname, skiprows=row_ctr, max_rows=1))

        row_ctr += 2

        self.num_Dof = int(np.loadtxt(fname, skiprows=row_ctr, max_rows=1))

        row_ctr += 2

        self.Dof = np.loadtxt(fname, skiprows=row_ctr, max_rows=1)

        row_ctr += 3

        self.mode_data = np.zeros(
            (self.num_modes, self.num_grid_points, self.num_Dof))

        # read eigen vectors
        for i in range(self.num_modes):
            self.mode_data[i, :, :] = np.loadtxt(
                fname, skiprows=row_ctr, max_rows=self.num_grid_points)
            row_ctr += self.num_grid_points + 1

        self.grid_points = np.loadtxt(
            fname, skiprows=row_ctr, max_rows=self.num_grid_points)

    def calculate_norms(self):
        self.norms = np.zeros((self.num_modes, self.num_grid_points))
        for i in range(self.num_modes):
            for j in range(self.num_grid_points):
                self.norms[i, j] = np.linalg.norm(self.mode_data[i, j, :])

    def write_grid_tec(self, fname):
        f = open(fname, "w")
        for i in range(self.num_grid_points):
            f.write("{}, {}, {}\n".format(
                self.grid_points[i, 0], self.grid_points[i, 1], self.grid_points[i, 2]))
        f.close()

    def write_grid_csv(self, fname: str):
        f = open(fname, 'w')
        f.write("X, Y, Z, ")
        for i in range(self.num_modes):
            f.write("m{}X, m{}Y, m{}Z, ".format(i, i, i))
        f.write("\n")
        # dump points file
        for i in range(self.num_grid_points):
            f.write('{}, {}, {}, '.format(
                self.grid_points[i, 0], self.grid_points[i, 1], self.grid_points[i, 2]))
            for j in range(self.num_modes):
                for k in range(3):
                    f.write('{}, '.format(self.mode_data[j, i, k]))
            f.write('\n')

        f.close()

    def calculate_mode_frequencies(self):
        self.mode_freqs = np.array(
            [np.sqrt(i) for i in self.eigenvalues])

    def write_modes(self, fname='modes.cba'):
        with open(fname, 'w') as f:
            f.write('1\n')
            f.write('Number of Modes\n')
            f.write('{}\n'.format(self.num_modes))
            f.write('eigenvalues:\n')
            for e in self.eigenvalues:
                f.write('{}\n'.format(e))
            f.write('PLT1\n')
            f.write('{}\n'.format(self.num_grid_points))
            f.write('Numer of Dof\n')
            f.write('{}\n'.format(self.num_modes))
            f.write('degrees of freedom (x=1, y=2, z=3, rotx=4, roty=5, rotz=6)\n')
            for d in self.Dof:
                f.write('{} '.format(int(d)))
            f.write('\n')
            f.write('eigenvectors: \n')
            for m in range(self.num_modes):
                f.write('{}\n'.format(m))
                for i in range(self.num_grid_points):
                    for j in range(self.num_modes):
                        f.write('{} \t'.format(self.mode_data[m, i, j]))
                    f.write('\n')
            f.write('grid\n')
            for i in range(self.num_grid_points):
                for j in range(3):
                    f.write('{} \t'.format(self.grid_points[i, j]))
                f.write('\n')


class NastranReader:
    def __init__(self, filename):
        self.filename = filename
        print(
            " Reading Nastran data from {}".format(self.filename))
        self.op2 = read_op2(filename + ".op2",
                            build_dataframe=False, debug=False)
        self.bdf = BDF(debug=False)

    def read_grid_points(self):
        print(" Reading grid points from %s" %
              self.filename + ".bdf")
        self.bdf.read_bdf(self.filename + ".bdf")

        count = 0
        self.structural_nodes = set()

        mass_elements = ['CONM1', 'CONM2',
                         'CMASS1', 'CMASS2', 'CMASS3', 'CMASS4']

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
        print(" Reading %i loadcase(s) from %s" %
              (num_cases, self.filename + ".op2"))

        if num_cases > 1:
            print(
                " More than one load case in %s" % self.filename)

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
        print(" Modal frequencies (Hz) %s" % " ".join(
            ['{: .2f}'.format(freq / (2.0 * np.pi)) for freq in self.mode_freqs]))

        self.mode_data = mode_data

    def print_op2_stats(self, short=False):
        print(self.op2.get_op2_stats(short), flush=True)

    def print_bdf(self):
        print(self.bdf.object_attributes(), flush=True)
        print(self.bdf.object_methods(), flush=True)
