import numpy as np

class genericmodalmodel():
    def __init__(self, grid_points, mode_freqs, mode_data, mode_list=None, mode_damping=None):
        self.grid_points = grid_points
        self.mode_freqs = mode_freqs
        self.mode_data = mode_data
        self.num_grid_points = grid_points.shape[0]
        self.num_modes = len(mode_freqs)
        self.modal_displacements = [0 for i in range(self.num_modes)]
        if not mode_list:
            self.mode_list = [i for i in range(self.num_modes)]
        else: 
            self.mode_list = mode_list
            self.num_modes = len(self.mode_list)

        if not mode_damping:
            self.mode_damping = [0.0 for i in range(self.num_modes)]
        else:
            self.mode_damping = mode_damping
         
    def calculate_modal_forcing(self, force):
        modal_forcing = np.zeros((self.num_modes, 1))
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
    
            
