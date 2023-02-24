import numpy as np

class genericmodalmodel():
    def __init__(self, grid_points, mode_freqs, mode_data, mode_list=None, mode_damping=None):
        self.grid_points = grid_points
        self.mode_freqs = mode_freqs
        self.mode_data = mode_data
        self.num_grid_points = grid_points.shape[0]
        self.num_modes = len(mode_freqs)
        self.critical_damping = 2.0 * self.mode_freqs
        if not mode_list:
            self.mode_list = [i for i in range(self.num_modes)]
        else: 
            self.mode_list = mode_list
            self.num_modes = len(self.mode_list)

        if not mode_damping:
            self.mode_damping = np.array([0.0 for i in range(self.num_modes)])
        else:
            self.mode_damping = mode_damping

        # modal perturbations at TimeN

        self.modal_displacementsTn = np.zeros(self.num_modes)
        self.modal_velocitiesTn = np.zeros(self.num_modes)
        self.modal_accelerationsTn = np.zeros(self.num_modes)

        # modal perturbations at TimeNPlus1

        self.modal_displacementsTnP1 = np.zeros(self.num_modes)
        self.modal_velocitiesTnP1 = np.zeros(self.num_modes)
        self.modal_accelerationsTnP1 = np.zeros(self.num_modes)

        self.newmark_alpha = 0.25
        self.newmark_delta = 0.5
         
    def calculate_modal_forcing(self, force):
        modal_forcing = np.zeros((self.num_modes, 1))
        f_t = np.sum(force, axis=0)
        for i, m in enumerate(self.mode_list):
            for j in range(self.num_grid_points):
                for k in range(3):
                    modal_forcing[i] += self.mode_data[m, j, k] * force[j, k]
        return modal_forcing

    def calculate_modal_displacements(self, modal_forcing, relaxation_factor):
        displacements = np.zeros_like(self.grid_points)
        for i, m in enumerate(self.mode_list):
            self.modal_displacements[m] = self.modal_displacements[m] + relaxation_factor * (modal_forcing[i] - self.mode_freqs[m] ** 2 * self.modal_displacements[m])
            for j in range(self.num_grid_points):
                for k in range(3):
                    displacements[j, k] += self.modal_displacements[m] * self.mode_data[m, j, k]
        return displacements

    def march_modal_model(self, modal_forcing, dt):
        displacements = np.zeros_like(self.grid_points)

        for m in range(self.num_modes):
            l = self.mode_freqs[m] ** 2
            a_ii = 1.0 + dt * dt * self.newmark_alpha * l + 2.0 * dt * self.newmark_delta * self.mode_damping[m] * self.mode_freqs[m]
            b_ii = modal_forcing[m] - (2.0 * self.mode_damping[m] * self.mode_freqs[m] * (self.modal_accelerationsTn[m] + dt * (1.0 - self.newmark_delta) * self.modal_accelerationsTn[m])) - (l * (self.modal_displacementsTn[m] + dt * self.modal_velocitiesTn[m] + dt * dt * (0.5 - self.newmark_alpha) * self.modal_accelerationsTn[m]))

            self.modal_accelerationsTnP1[m] = b_ii / a_ii

            self.modal_velocitiesTnP1[m] = self.modal_velocitiesTn[m] + dt * (1.0 - self.newmark_delta) * self.modal_accelerationsTn[m] + dt * self.newmark_delta * self.modal_accelerationsTnP1[m]
            self.modal_displacementsTnP1[m] = self.modal_displacementsTn[m] + dt * self.modal_velocitiesTn[m] + dt * dt * (0.5 - self.newmark_alpha) * self.modal_accelerationsTn[m] + self.newmark_alpha * dt * dt * self.modal_accelerationsTnP1[m]

            displacements += self.modal_displacementsTnP1[m] * self.mode_data[m, :, 0:3]

        self.copy_time_history()

        return displacements
    
    def flatten_modes(self):
        self.flat_modes = np.zeros((self.num_grid_points * 3, self.num_modes))
        for i in range(self.num_grid_points):
            for j in range(3):
                for k in range(self.num_modes):
                    self.flat_modes[i * 3 + j, k] = self.mode_data[k, i, j]

    def solve_flat_modes(self, force: np.array):
        flat_force = force.flatten()
        modal_forcing = self.flat_modes.T @ flat_force
        return modal_forcing

    def copy_time_history(self):
        self.modal_displacementsTn = self.modal_accelerationsTnP1.copy()
        self.modal_velocitiesTn = self.modal_velocitiesTnP1.copy()
        self.modal_accelerationsTn = self.modal_accelerationsTnP1.copy()

        self.modal_displacementsTn = np.zeros_like(self.modal_displacementsTnP1)
        self.modal_velocitiesTn = np.zeros_like(self.modal_velocitiesTnP1)
        self.modal_accelerationsTn = np.zeros_like(self.modal_accelerationsTnP1)

    def write_grid_csv(self, fname='modal_model_grid.csv'):
        with open(fname, 'w') as f:
            f.write('X, Y, Z, ')
            for m in range(self.num_modes):
                f.write('mode{}X, mode{}Y, mode{}Z, '.format(m, m, m))
            f.write('\n')
            for i in range(self.num_grid_points):
                for j in range(3):
                    f.write('{}, '.format(self.grid_points[i, j]))
                for m in range(self.num_modes):
                    f.write()