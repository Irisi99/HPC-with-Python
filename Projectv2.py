import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema

s = 4
x_n = y_n = 50
time_steps = 3000
omega = 1
eps = 0.01
rho_0 = 1
wall_velocity = 0.1
wall = 'top'
inlet_rho = 1.005
outlet_rho = 1
flow_direction = 'right'

while len(sys.argv) >= s:
    match sys.argv[s-2]:
        case '-n':
            x_n = y_n = int(sys.argv[s-1])
        case '-t':
            time_steps = int(sys.argv[s-1])
        case '-w':
            omega = float(sys.argv[s-1])
        case '-e':
            eps = float(sys.argv[s-1])
        case '-d':
            rho_0 = float(sys.argv[s-1])
        case '-v':
            wall_velocity = float(sys.argv[s-1])
        case '-i':
            inlet_rho = float(sys.argv[s-1])
        case '-o':
            outlet_rho = float(sys.argv[s-1])
        case '-m':
            if sys.argv[s-1] != 'top' and sys.argv[s-1] != 'bottom':
                print('Orientation of moving wall can only be \'top\' or \'bottom\'')
                sys.exit()
            else:
                wall = sys.argv[s-1]
        case '-f':
            if sys.argv[s-1] != 'left' and sys.argv[s-1] != 'right':
                print('Direction of flow can only be \'left\' or \'right\'')
                sys.exit()
            else:
                flow_direction = sys.argv[s-1]
    s += 2

delta_rho = outlet_rho - inlet_rho
c_s = (1/np.sqrt(3))

c = np.array([
    [0, 1, 0, -1, 0, 1, -1, -1, 1],
    [0, 0, 1, 0, -1, 1, 1, -1, -1]
]).T

w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

W = np.arange(0.1, 2, 0.1)
analytical_viscosity = []
simulated_viscosity = []


def stream(f):
    for i in range(1, 9):
        f[i] = np.roll(f[i], c[i], axis=(0, 1))


def equilibrium(rho, u):
    cu = np.dot(u.T, c.T).T
    uu = np.sum(u**2, axis=0)
    return (w*(rho*(1 + 3*cu + 9/2*cu**2 - 3/2*uu)).T).T


def collide(f, omega):
    rho = np.einsum('xyz->yz', f)
    u = np.einsum('xyz,xi->iyz', f, c) / rho
    f += omega*(equilibrium(rho, u) - f)
    return rho, u


def stokes_condition(omega):

    points = []

    if sys.argv[1] == 'shear_wave_decay_density':
        u = np.zeros((2, x_n, y_n))
        rho = np.zeros((x_n, y_n))
        x = np.linspace(0, x_n, x_n)
        for i in np.arange(y_n):
            rho[:, i] = rho_0 + eps*np.sin(2*np.pi*x/x_n)
    else:
        rho = np.ones((x_n, y_n))
        u = np.zeros((2, x_n, y_n))
        y = np.linspace(0, y_n, y_n)
        for i in np.arange(x_n):
            u[0][i, :] = eps*np.sin(2*np.pi*y/y_n)

    f = equilibrium(rho, u)

    for t in range(time_steps):
        stream(f)
        rho, u = collide(f, omega)
        if sys.argv[1] == 'shear_wave_decay_density':
            points.append(np.max(np.abs(rho - rho_0)))
        else:
            points.append(np.max(np.abs(u[0])))

    if sys.argv[1] == 'shear_wave_decay_density':
        size = x_n
        stokes_points = np.array(points)
        x = argrelextrema(stokes_points, np.greater)
        stokes_points = stokes_points[x]
        x = np.array(x).squeeze()

    else:
        size = y_n
        stokes_points = np.array(points)
        x = np.arange(0, time_steps)

    value = curve_fit(lambda t, v: eps * np.exp(-v *
                      (2 * np.pi / size) ** 2 * t), x, stokes_points)[0][0]

    analytical_viscosity.append(1/3 * (1/omega - 0.5))
    simulated_viscosity.append(value)


def rigidWall(wall):
    if wall == 'top':
        f[[4, 7, 8], :, 0] = f[[2, 5, 6], :, 0]
    elif wall == 'bottom':
        f[[2, 5, 6], :, y_n-1] = f[[4, 7, 8], :, y_n-1]
    elif wall == 'right':
        f[[3, 7, 6], x_n-1] = f[[1, 5, 8], x_n-1]
    else:
        f[[1, 5, 8], 0] = f[[3, 7, 6], 0]


def movingWall():
    avg_rho = np.mean(rho)
    multiplier = 2 * avg_rho * w / c_s**2

    if wall == 'top':
        f[4, :, y_n-1] = f[2, :, y_n-1] - multiplier[2] * c[2] @ wall_velocity
        f[7, :, y_n-1] = f[5, :, y_n-1] - multiplier[5] * c[5] @ wall_velocity
        f[8, :, y_n-1] = f[6, :, y_n-1] - multiplier[6] * c[6] @ wall_velocity
    else:
        f[2, :, 0] = f[4, :, 0] - multiplier[4] * c[4] @ wall_velocity
        f[5, :, 0] = f[7, :, 0] - multiplier[7] * c[7] @ wall_velocity
        f[6, :, 0] = f[8, :, 0] - multiplier[8] * c[8] @ wall_velocity


def flow():
    rho_in = np.ones((x_n)) * inlet_rho / c_s**2
    rho_out = np.ones((x_n)) * outlet_rho / c_s**2

    if flow_direction == 'right':
        f_in = equilibrium(rho_in, u[:, x_n-1, :])
        f_out = equilibrium(rho_out, u[:, 0, :])
        f_eq = equilibrium(rho, u)

        inlet = f_in + (f[:, x_n-1] - f_eq[:, x_n-1])
        outlet = f_out + (f[:, 0] - f_eq[:, 0])

        f[[1, 5, 8], 0] = inlet[[1, 5, 8]]
        f[[3, 6, 7], x_n-1] = outlet[[3, 6, 7]]

    else:
        f_in = equilibrium(rho_in, u[:, 0, :])
        f_out = equilibrium(rho_out, u[:, x_n-1, :])
        f_eq = equilibrium(rho, u)

        inlet = f_in + (f[:, 0] - f_eq[:, 0])
        outlet = f_out + (f[:, x_n-1] - f_eq[:, x_n-1])

        f[[3, 6, 7], x_n-1] = inlet[[3, 6, 7]]
        f[[1, 5, 8], 0] = outlet[[1, 5, 8]]


if sys.argv[1] == "shear_wave_decay_density":
    fig, ax = plt.subplots()
    u = np.zeros((2, x_n, y_n))
    rho = np.zeros((x_n, y_n))
    x = np.linspace(0, x_n, x_n)
    for i in np.arange(y_n):
        rho[:, i] = rho_0 + eps*np.sin(2*np.pi*x/x_n)

    f = equilibrium(rho, u)

    avg_rho = []
    os.makedirs('./shear_wave_decay_density', exist_ok=True)

    for t in range(time_steps):
        stream(f)
        rho, u = collide(f, omega)
        avg_rho.append(rho[int(x_n/4), int(y_n/2)])
        if t % 150 == 0 and t <= 1050:
            print('We are at timestep {}'.format(t))
            ax.set_ylim([rho_0 - eps, rho_0 + eps])
            ax.set_ylabel('density rho at y = '+str(int(y_n/2)))
            ax.set_xlabel('x position')
            ax.set_title('t = '+str(t))
            ax.plot(rho[:, int(y_n/2)])

    fig.savefig('shear_wave_decay_density/' + str(t) +
                'steps.png', bbox_inches='tight')

    plt.clf()
    plt.ylim([rho_0 - eps, rho_0 + eps])
    plt.ylabel('density rho at x = '+str(int(x_n/2))+' y = '+str(int(y_n/2)))
    plt.xlabel('timestep t')
    plt.plot(avg_rho)
    plt.savefig('shear_wave_decay_density/evouliton_over_time.png',
                bbox_inches='tight')

    for x in W:
        stokes_condition(x)

    plt.clf()
    plt.yscale('log')
    plt.ylabel('velocity v')
    plt.xlabel('relaxation term w')
    plt.plot(W, analytical_viscosity, color='orange',
             label='analytic viscosity')
    plt.plot(W, simulated_viscosity, color='cornflowerblue',
             label='simulated viscosity')
    plt.legend()
    plt.savefig('shear_wave_decay_density/w_relaxation.png',
                bbox_inches='tight')


elif sys.argv[1] == "shear_wave_decay_velocity":
    fig, ax = plt.subplots()
    rho = np.ones((x_n, y_n))
    u = np.zeros((2, x_n, y_n))
    y = np.linspace(0, y_n, y_n)
    for i in np.arange(x_n):
        u[0][i, :] = eps*np.sin(2*np.pi*y/y_n)

    f = equilibrium(rho, u)

    avg_u = []
    os.makedirs('./shear_wave_decay_velocity', exist_ok=True)

    for t in range(time_steps):
        stream(f)
        rho, u = collide(f, omega)
        avg_u.append(u[0, int(x_n/2), int(y_n/4)])
        if t % 150 == 0 and t <= 1050:
            print('We are at timestep {}'.format(t))
            ax.set_ylim([-eps, +eps])
            ax.set_ylabel('velocity u at x = '+str(int(x_n/2)))
            ax.set_xlabel('y position')
            ax.set_title('t = '+str(t))
            ax.plot(u[0, int(x_n/2)])

    fig.savefig('shear_wave_decay_velocity/' + str(t) +
                'steps.png', bbox_inches='tight')

    plt.clf()
    plt.ylim([0, eps])
    plt.ylabel('velocity u at x = '+str(int(x_n/2))+' y = '+str(int(y_n/4)))
    plt.xlabel('timestep t')
    plt.plot(avg_u)
    plt.savefig('shear_wave_decay_velocity/evouliton_over_time.png',
                bbox_inches='tight')

    for x in W:
        stokes_condition(x)

    plt.clf()
    plt.yscale('log')
    plt.ylabel('velocity v')
    plt.xlabel('relaxation term w')
    plt.plot(W, analytical_viscosity, color='orange',
             label="analytic viscosity")
    plt.plot(W, simulated_viscosity, color='cornflowerblue',
             label="simulated viscosity")
    plt.legend()
    plt.savefig('shear_wave_decay_velocity/w_relaxation.png')


elif sys.argv[1] == "couette_flow":
    rho = np.ones((x_n, y_n))
    u = np.zeros((2, x_n, y_n))
    f = equilibrium(rho, u)

    y = np.arange(y_n)
    wall_velocity = [wall_velocity, 0.0]

    if wall == 'top':
        analytical_viscosity = (y) / (y_n-1) * wall_velocity[0]
    else:
        analytical_viscosity = (y_n-1 - y) / (y_n-1) * wall_velocity[0]

    os.makedirs('./couette_flow', exist_ok=True)
    plt.ylim([-1, y_n])
    plt.ylabel('y position')
    plt.xlabel('velocity u at x = '+str(int(x_n/2)))

    for t in range(time_steps):
        stream(f)
        movingWall()
        rigidWall(wall)
        rho, u = collide(f, omega)

        if t == 0:
            plt.plot(u[0, int(x_n/2)], y, color='cornflowerblue',
                     label='Simulated viscosity')
        elif t % 200 == 0:
            plt.plot(u[0, int(x_n/2)], y, color='cornflowerblue')

    plt.plot(analytical_viscosity, y, color='orange',
             label='Analytical viscosity')

    plt.axhline(y=-0.5 if wall == 'top' else y_n-0.5,
                color='black', label='Rigid Wall')
    plt.axhline(y=y_n-0.5 if wall == 'top' else -
                0.5, color='red', label='Moving Wall')

    plt.legend()
    plt.savefig('couette_flow/couette_flow_'+wall+'_t=' +
                str(time_steps)+'.png', bbox_inches='tight')


elif sys.argv[1] == "poiseuille_flow":
    rho = np.ones((x_n, y_n))
    u = np.zeros((2, x_n, y_n))
    f = equilibrium(rho, u)

    os.makedirs('./poiseuille_flow', exist_ok=True)
    plt.ylim([-1, y_n])
    plt.ylabel('y position')
    plt.xlabel('velocity u at x = '+str(int(x_n/2)))

    y = np.arange(y_n)
    v = 1/3 * (1/omega - 0.5)
    derivative = c_s**2 * delta_rho / (x_n * 2)
    avg_rho = np.mean(rho)

    if flow_direction == 'right':
        analytical_viscsity = -1/(2*avg_rho*v)*derivative*y*(y_n-1-y)
    else:
        analytical_viscsity = 1/(2*avg_rho*v)*derivative*y*(y_n-1-y)

    for t in range(time_steps):
        flow()
        stream(f)
        rigidWall('top')
        rigidWall('bottom')
        rho, u = collide(f, omega)
        if t == 0:
            plt.plot(u[0, int(x_n/2)], y, color='cornflowerblue',
                     label='Simulated viscosity')
        elif t % 200 == 0:
            plt.plot(u[0, int(x_n/2)], y, color='cornflowerblue')

    plt.plot(analytical_viscsity, y, color='orange',
             label='Analytical viscosity')

    plt.axhline(y=-0.5, color='black', label='Rigid Wall')
    plt.axhline(y=y_n-0.5, color='black')

    plt.legend()
    plt.savefig('poiseuille_flow/poiseuille_flow_'+str(flow_direction)+'_t=' +
                str(time_steps)+'.png', bbox_inches='tight')


elif sys.argv[1] == "sliding_lid":
    rho = np.ones((x_n, y_n))
    u = np.zeros((2, x_n, y_n))
    f = equilibrium(rho, u)

    os.makedirs('./sliding_lid', exist_ok=True)

elif sys.argv[1] == "sliding_lid_parallelized":
    rho = np.ones((x_n, y_n))
    u = np.zeros((2, x_n, y_n))
    f = equilibrium(rho, u)

    os.makedirs('./sliding_lid_parallelized', exist_ok=True)

else:
    print('Invalid Argument! Please use one of the proper arguments below:')
    print('- shear_wave_decay_density')
    print('- shear_wave_decay_velocity')
    print('- couette_flow')
    print('- poiseuille_flow')
    print('- sliding_lid')
    print('- sliding_lid_parallelized\n')

    print('Optional Arguments include:')
    print('-n (int) : Grid Size')
    print('-t (int) : Number of Iterations')
    print('-w (float) : Omega')
    print('-e (float) : Epsilon')
    print('-d (float) : Initial Density')
    print('-v (float) : Sliding Wall Velocity')
    print(
        '-m (string) {\'top\',\'bottom\'} : Decides which of the walls moves')
    print('-i (float) : Inlet Density')
    print('-o (float) : Outlet Density')
    print(
        '-f (string) {\'left\', \'right\'} : Decides which way the flow goes')
