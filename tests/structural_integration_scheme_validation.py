import numpy as np
import matplotlib.pyplot as plt


f = 2
time = 10
samples = 2000

m = 8
c = 0
k = 50
A0 = 5

omega =np.sqrt(k/m)
delta = c /(2*m)
omega_d = np.sqrt(omega**2 - delta**2)

t = np.linspace(0, time, samples)

phi = 0

x = np.exp(-delta * t) * (np.cos(omega_d * t + phi))
dx = -np.exp(-delta * t) * (delta * np.cos(omega_d * t) + omega * np.sin(omega_d * t))
ddx = np.exp(-delta * t) * ((delta ** 2 - omega_d ** 2) * np.cos(omega_d * t) + 2 * omega * delta * np.sin(omega_d * t))

force = 0

alpha = 0.25
delta = 0.50

x0 = x[0]
v0 = 0
a0 = 0


dt = 0.1

nsteps = int(time / dt)

fig, ax = plt.subplots(3, 1, constrained_layout=True)

ax[0].plot(t, x)
ax[1].plot(t, dx)
ax[2].plot(t, ddx)
# perform newmark integration

time = 0

for i in range(nsteps):
    A = force - c * (a0 + dt * (1-delta) * a0) - k*(x0 + dt * v0 + dt * dt * (0.5 - alpha) * a0)
    B = m + dt * dt * alpha * k + dt * delta * c

    a1 = A/B
    v1 = v0 + dt * (1 - delta) * a0 + dt * delta * a1
    x1 = x0 + dt * v0 + dt * dt * (0.5 - alpha) * a0 + alpha * dt * dt * a1

    print(v1) 

    ax[0].plot(time, x1, 'rx', markersize=5)
    ax[1].plot(time, v1, 'rx', markersize=5)
    ax[2].plot(time, a1, 'rx', markersize=5)
    
    time += dt


    a0 = a1
    v0 = v1
    x0 = x1

# dt = 0.2

# nsteps = int(time / dt)
# time=0
# for i in range(nsteps):
#     A = force - c * (a0 + dt * (1-delta) * a0) - k*(x0 + dt * v0 + dt * dt * (0.5 - alpha) * a0)
#     B = m + dt * dt * alpha * k + dt * delta * c

#     a1 = A/B
#     v1 = v0 + dt * (1 - delta) * a0 + dt * delta * a1
#     x1 = x0 + dt * v0 + dt * dt * (0.5 - alpha) * a0 + alpha * dt * dt * a1

#     print(v1) 

#     ax[0].plot(time, x1, 'kx', markersize=5)
#     ax[1].plot(time, v1, 'kx', markersize=5)
#     ax[2].plot(time, a1, 'kx', markersize=5)
    
#     time += dt


#     a0 = a1
#     v0 = v1
#     x0 = x1

ax[0].set_title('Displacement', fontweight='bold', fontsize=20)
ax[1].set_title('Velocity', fontweight='bold', fontsize=20)
ax[2].set_title('Acceleration', fontweight='bold', fontsize=20)
fig.set_figheight(8)
fig.set_figwidth(8)

ax[2].legend(['Analytical Solution','Newmark Integration'], bbox_to_anchor=[0.5, -0.75], ncol=2, loc='center', fontsize=15)

for a in ax:
    a.tick_params(labelsize=20)

ax[0].set_ylabel('$x(t)$', fontweight='bold', fontsize=20)
ax[1].set_ylabel('$x\'(t)$', fontweight='bold', fontsize=20)
ax[2].set_ylabel('$x\'\'(t)$', fontweight='bold', fontsize=20)

ax[2].set_xlabel('$t$', fontweight='bold', fontsize=20)

# fig.tight_layout()


fig.savefig('newmark.eps', format='eps')
fig.savefig('newmark.png', format='png')
