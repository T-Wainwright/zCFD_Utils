import numpy as np
from zcfd.utils import config
import os

"collection of integrators for FSI problems"


class Integrator_base():
    def __init__(self, num_modes, M, C, K) -> None:
        self.n_DOF = num_modes

        self.q = np.zeros((self.n_DOF, 1))
        self.qdot = np.zeros((self.n_DOF, 1))
        self.qddot = np.zeros((self.n_DOF, 1))

        self.q_n = np.zeros((self.n_DOF, 1))
        self.qdot_n = np.zeros((self.n_DOF, 1))
        self.qddot_n = np.zeros((self.n_DOF, 1))

        self.M = M
        self.C = C
        self.K = K

    def integrate(self, F):
        pass

    def copy_time_history(self):
        # copy current time vector to previous timestep
        self.q_n = self.q.copy()
        self.qdot_n = self.qdot.copy()
        self.qddot_n = self.qddot.copy()

        # reset current time vector
        self.q = np.zeros_like(
            self.q)
        self.qdot = np.zeros_like(self.qdot)
        self.qddot = np.zeros_like(
            self.qddot)

    def set_initial_conditions(self, **kwargs):
        f0 = np.zeros((self.n_DOF, 1))
        if 'q0' in kwargs:
            q0 = kwargs.get('q0')
            for m in range(self.n_DOF):
                self.q[m] = q0[m]
                self.q_n[m] = q0[m]

        if 'f0' in kwargs:
            f0 = np.array(kwargs.get('f0'))

        RHS = np.zeros((self.n_DOF, 1))
        RHS -= self.C.dot(self.qdot)
        RHS -= self.K.dot(self.q)

        self.qddot = np.linalg.solve(self.M, RHS)
        self.qddot_n = self.qddot

    def write_force_history(self, solve_cycle, real_timestep, force, fname='modal_force_history.csv'):
        if os.path.exists(fname):
            f = open(fname, 'a')
        else:
            f = open(fname, 'w')
            f.write('Solve_Cycle, Real_TimeStep, ')
            for m in range(self.n_DOF):
                f.write('mode{}_Weight, mode{}_displacement, mode{}_velocity, mode{}_acceleration, '.format(
                    m, m, m, m))
            f.write('\n')
        f.write('{}, {}, '.format(solve_cycle, real_timestep))
        for m in range(self.n_DOF):
            f.write('{}, {}, {}, {},'.format(
                force[m], self.q[m], self.qdot[m], self.qddot[m]))
        f.write('\n')
        f.close()


class Generalised_a_Integrator(Integrator_base):
    def __init__(self, num_modes, M, C, K, dt, rho_e=0.5) -> None:
        super().__init__(num_modes, M, C, K)

        self.a = np.zeros((self.n_DOF, 1))
        self.a_n = np.zeros((self.n_DOF, 1))
        self.rho_e = rho_e
        self.dt = dt

        self.set_integration_parameters()

    def set_integration_parameters(self):
        self.alpha_m = (2.0*self.rho_e-1.0)/(self.rho_e+1.0)
        self.alpha_f = (self.rho_e)/(self.rho_e+1.0)
        self.gamma = 0.5+self.alpha_f-self.alpha_m
        self.beta = 0.25*(self.gamma+0.5)**2

        self.gammaPrime = self.gamma/(self.dt*self.beta)
        self.betaPrime = (1.0-self.alpha_m)/((self.dt**2)
                                             * self.beta*(1.0-self.alpha_f))

    def integrate(self, F):
        eps = 1e-6

        self.a += (self.alpha_f) / (1 - self.alpha_m) * self.qddot_n
        self.a -= (self.alpha_m) / (1 - self.alpha_m) * self.a_n

        self.q = np.copy(self.q_n)
        self.q += self.dt * self.qdot_n
        self.q += (0.5 - self.beta) * self.dt * self.dt * self.a_n
        self.q += self.dt * self.dt * self.beta * self.a

        self.qdot = np.copy(self.qdot_n)
        self.qdot += (1 - self.gamma) * self.dt * self.a_n
        self.qdot += self.dt * self.gamma * self.a

        res = self.compute_residual(F)

        while (np.linalg.norm(res) >= eps):
            St = self.tangent_operator()
            Deltaq = -1 * (np.linalg.solve(St, res))
            self.q += Deltaq
            self.qdot += self.gammaPrime * Deltaq
            self.qddot += self.betaPrime * Deltaq
            res = self.compute_residual(F)

        self.a += (1 - self.alpha_f) / (1 - self.alpha_m) * self.qddot

    def compute_residual(self, F):
        return self.M.dot(self.qddot) + self.C.dot(self.qdot) + self.K.dot(self.q) - F

    def tangent_operator(self):
        return self.betaPrime * self.M + self.gammaPrime * self.C + self.K

    def copy_time_history(self):
        super().copy_time_history()

        self.a_n = self.a.copy()
        self.a = np.zeros_like(self.a)


class Newmark_Integrator(Integrator_base):
    def __init__(self, num_modes, M, C, K, dt, alpha=0.25, delta=0.5) -> None:
        super().__init__(num_modes, M, C, K)

        self.newmark_alpha = alpha
        self.newmark_delta = delta

        self.C *= 2.0 * np.sqrt(K)

        self.dt = dt

    def integrate(self, F):
        # diaglonalise matrices for easier calculation
        q_n = np.diag(self.q_n)
        qdot_n = np.diag(self.qdot_n)
        qddot_n = np.diag(self.qddot_n)

        self.qddot = np.solve((self.M + self.dt * self.dt * self.newmark_alpha * K + self.dt * self.newmark_delta * self.C), (F - self.K @ (qddot_n + self.dt * (1 - self.newmark_delta) * qddot_n) - self.K @ (
            q_n + self.dt * qdot_n + self.dt * self.dt * (0.5 - self.newmark_alpha) * qddot_n)))

        self.qdot = qdot_n + self.dt * \
            (1 - self.newmark_delta) * self.dt * \
            self.newmark_delta * np.diag(self.qddot)
        self.q = q_n + self.dt * qdot_n + self.dt * self.dt * \
            (0.5 - self.newmark_alpha) * qddot_n + \
            self.newmark_alpha * self.dt * self.dt * np.diag(self.qddot)


class Relaxation_Integrator(Integrator_base):
    def __init__(self, num_modes, M, C, K, relaxation_factor=0.5) -> None:
        super().__init__(num_modes, M, C, K)

        self.relaxation_factor = relaxation_factor

    def integrate(self, F):
        print('flag')
        self.q = self.q_n + self.relaxation_factor * (F - self.q_n)