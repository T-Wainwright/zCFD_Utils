import numpy as np
from zcfd.utils import config
import os
import zcfdutils.integrators


class genericmodalmodel:
    def __init__(
        self,
        reader,
        dt,
        mode_list=None,
        mode_damping=None,
        integrator="generalised alpha",
        pitch=0,
    ):
        self.pitch = pitch
        # Set initial values
        self.grid_points = reader.grid_points @ z_rot(self.pitch)
        self.mode_freqs = np.array([i / (2 * np.pi) for i in reader.mode_freqs])
        self.mode_data = reader.mode_data
        self.num_grid_points = reader.grid_points.shape[0]
        self.num_modes = len(reader.mode_freqs)
        self.critical_damping = 2.0 * self.mode_freqs

        if not mode_list:
            self.mode_list = [i for i in range(self.num_modes)]
        else:
            self.mode_list = mode_list
            # self.num_modes = len(self.mode_list)
            # self.mode_freqs = np.array([reader.mode_freqs[i] / (2 * np.pi)  for i in mode_list])
            # self.mode_data = np.zeros((self.num_modes, self.num_grid_points, reader.mode_data.shape[2]))
            for i, m in enumerate(self.mode_list):
                self.mode_data[i, :, :] = reader.mode_data[m, :, :]

        if not mode_damping:
            self.mode_damping = np.array([0.0 for i in range(self.num_modes)])
        else:
            self.mode_damping = mode_damping

        self.modal_forcing = np.zeros((self.num_modes, 1))

        print(self.mode_data.shape)

        print(self.mode_data.shape)

        self.dt = dt

        self.set_integrator(integrator, reader)

    def set_integrator(self, integrator, reader):
        if integrator == "generalised alpha":
            self.integrator = zcfdutils.integrators.Generalised_a_Integrator(
                reader, self.dt
            )
        elif integrator == "newmark":
            self.integrator = zcfdutils.integrators.Newmark_Integrator(reader, self.dt)
        elif integrator == "relaxation":
            self.integrator = zcfdutils.integrators.Relaxation_Integrator(reader)
        elif integrator == "direct solve":
            self.integrator = zcfdutils.integrators.Direct_Solve(reader)

    def set_initial_conditions(self, initial):
        self.integrator.set_initial_conditions(q0=initial)

        displacements = np.zeros_like(self.grid_points)
        for m in self.mode_list:
            displacements += self.integrator.q[m] * self.mode_data[m, :, 0:3]

        self.write_force_history(0, 0)

        return displacements

    def calculate_modal_forcing(self, force):
        self.modal_forcing = np.zeros((self.num_modes, 1))

        for i in range(3):
            self.modal_forcing += np.array([self.mode_data[:, :, i].dot(force[:, i])]).T

    def integrate_solution(self):
        self.integrator.copy_time_history()
        self.integrator.integrate(self.modal_forcing)

    def write_force_history(self, solve_cycle, real_timestep):
        self.integrator.write_force_history(
            solve_cycle, real_timestep, self.modal_forcing, self.mode_list
        )

    def get_displacements(self):
        displacements = np.zeros((self.num_grid_points, 6))
        for m in self.mode_list:
            for i in range(self.num_grid_points):
                for j in range(6):
                    displacements[i, j] += (
                        self.integrator.q[m] * self.mode_data[m, i, j]
                    )
        return displacements

    def flatten_modes(self):
        self.flat_modes = np.zeros((self.num_grid_points * 3, self.num_modes))
        for i in range(self.num_grid_points):
            for j in range(3):
                for k in range(self.num_modes):
                    self.flat_modes[i * 3 + j, k] = self.mode_data[k, i, j]

    def solve_flat_modes(self, force: np.array):
        flat_force = force.flatten()
        self.modal_forcing = self.flat_modes.T @ flat_force
        return self.modal_forcing


def z_rot(t):
    t_rad = np.deg2rad(t)
    M = np.array(
        [
            [np.cos(t_rad), -np.sin(t_rad), 0],
            [np.sin(t_rad), np.cos(t_rad), 0],
            [0, 0, 1],
        ]
    )
    return M
