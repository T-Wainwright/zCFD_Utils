import numpy as np
import matplotlib.pyplot as plt

f = 2
time = 10
samples = 2000

m = 8
c = 11
k = 4000
A0 = 1

omega =np.sqrt(k)
zeta = c / (2 * np.sqrt(m * k))
omega_d = omega * np.sqrt(1 - zeta**2)

t = np.linspace(0, time, samples)

phi = 0

x = A0 * np.exp(-zeta * omega *t) * np.cos(omega_d * t - phi)


force = 0



alpha = 0.25
delta = 0.50

x0 = x[0]
v0 = 0
a0 = 0


dt = t[1] - t[0]

fig, ax = plt.subplots(3, 1)

ax[0].plot(t, x)
# perform newmark integration

for i in range(samples):
    A = force - c * (a0 + dt * (1-delta) * a0) - k*(x0 + dt * v0 + dt * dt * (0.5 - alpha) * a0)
    B = m + dt * dt * alpha * k + dt * delta * c

    a1 = A/B
    v1 = v0 + dt * (1 - delta) * a0 + dt * delta * a1
    x1 = x0 + dt * v0 + dt * dt * (0.5 - alpha) * a0 + alpha * dt * dt * a1

    ax[0].plot(t[i], x1, 'rx', markersize=1)
    ax[1].plot(t[i], v1, 'rx', markersize=1)
    ax[2].plot(t[i], a1, 'rx', markersize=1)


    a0 = a1
    v0 = v1
    x0 = x1

ax[0].set_title('Displacement')
ax[1].set_title('Velocity')
ax[2].set_title('Acceleration')
fig.tight_layout()
