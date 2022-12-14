import numpy as np

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

        self.n_pts = int(np.loadtxt(fname, skiprows=row_ctr, max_rows=1))

        row_ctr += 2

        self.n_Dof = int(np.loadtxt(fname, skiprows=row_ctr, max_rows=1))

        row_ctr += 2

        self.Dof = np.loadtxt(fname, skiprows=row_ctr, max_rows=1)

        row_ctr += 3

        self.eigenvectors = np.zeros((self.num_modes, self.n_pts, 3))

        # read eigen vectors
        for i in range(self.num_modes):
            self.eigenvectors[i, :, :] = np.loadtxt(
                fname, skiprows=row_ctr, max_rows=self.n_pts)
            row_ctr += self.n_pts + 1

        self.grid = np.loadtxt(fname, skiprows=row_ctr, max_rows=self.n_pts)

    def write_grid_tec(self, fname):
        f = open(fname, "w")
        for i in range(self.n_pts):
            f.write("{}, {}, {}\n".format(
                self.grid[i, 0], self.grid[i, 1], self.grid[i, 2]))
        f.close()

    def write_grid_csv(self, fname):
        f = open(fname, 'w')
        f.write("X, Y, Z, ")
        for i in range(self.num_modes):
            f.write("m{}X, m{}Y, m{}Z, ".format(i, i, i))
        f.write("\n")
        # dump points file
        for i in range(self.n_pts):
            f.write('{}, {}, {}, '.format(
                self.grid[i, 0], self.grid[i, 1], self.grid[i, 2]))
            for j in range(self.num_modes):
                for k in range(3):
                    f.write('{}, '.format(self.eigenvectors[j, i, k]))
            f.write('\n')

        f.close()

    def calculate_mode_frequencies(self):
        self.mode_frequencies = np.array(
            [np.sqrt(i) for i in self.eigenvalues])
