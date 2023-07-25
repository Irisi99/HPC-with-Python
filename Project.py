import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema

s = 4  # helper variable used to parse arguments given to the program by the user
x_n = y_n = 50  # default grid size set to 50x50
time_steps = 3000  # default time steps executed
omega = 1  # default relaxation parameter
eps = 0.01  # initial epsilon of shear wave decay
rho_0 = 1  # inital density for shear wave density decay
# default wall velocity for couette, poiseuille and sliding lid simulations
wall_velocity = 0.1
wall = 'top'  # specifiec which one of the walls moves in the couette simulation
inlet_rho = 1.005  # inlet flow in posieuille simulation
outlet_rho = 1  # outlet flow in posieuille simulation
flow_direction = 'right'  # direction of the flow in posieuille simulation
Re = 100  # initial Reynolds number


# We check how many properties were passed to the program and assign the above properties to their specified values
while len(sys.argv) >= s:
    match sys.argv[s-2]:
        case '-xn':
            x_n = int(sys.argv[s-1])
        case '-yn':
            y_n = int(sys.argv[s-1])
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
        case '-r':
            if int(sys.argv[s-1]) > 1000 or int(sys.argv[s-1]) <= 0:
                print('Reynolds number needs to be bigger than 0 and smaller than 1000')
                sys.exit()
            Re = int(sys.argv[s-1])

    s += 2

delta_rho = 1/2 * (outlet_rho - inlet_rho)  # the flow in posieuille simulation
c_s = 1/np.sqrt(3)   # latice units

# 9 directions of the velocity in the grid
c = np.array([
    [0, 1, 0, -1, 0, 1, -1, -1, 1],
    [0, 0, 1, 0, -1, 1, 1, -1, -1]
]).T

# the weights of every velocity direction (summ up to 1)
w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

# all the omega values we use to plot the relaxation of the shear wave decay (from 0.1 to 2 with step 0.1)
W = np.arange(0.1, 2, 0.1)

# variables used in the program initalized here for consistency
analytical_viscosity = []
simulated_viscosity = []


def stream(f):
    # stream funciton rolls all of the 9 dimensions of the velocity in the grid
    # elements that roll beyond the last position are re-introduced at the first position
    for i in range(1, 9):
        f[i] = np.roll(f[i], c[i], axis=(0, 1))


def equilibrium(rho, u):
    # equilibrium function generates the distribution array from the density and velocity
    # this is done via the equilibrium distribution equation
    cu = np.dot(u.T, c.T).T
    uu = np.sum(u**2, axis=0)
    return (w*(rho*(1 + 3*cu + 9/2*cu**2 - 3/2*uu)).T).T


def collide(f, omega):
    # collide function generates density and velocity from the distribution array
    # this is done by implying that the distribution array locally relaxes to an equilibrium distribution
    rho = np.einsum('xyz->yz', f)
    u = np.einsum('xyz,xi->iyz', f, c) / rho
    f += omega*(equilibrium(rho, u) - f)
    return rho, u


def stokes_condition(omega):
    points = []

    if sys.argv[1] == 'shear_wave_decay_density':
        # initialize the density shear wave decay density and velocity the same as in the main function
        u = np.zeros((2, x_n, y_n))
        rho = np.zeros((x_n, y_n))
        x = np.linspace(0, x_n, x_n)
        for i in np.arange(y_n):
            rho[:, i] = rho_0 + eps*np.sin(2*np.pi*x/x_n)
    else:
        # initialize the velocity shear wave decay density and velocity the same as in the main function
        rho = np.ones((x_n, y_n))
        u = np.zeros((2, x_n, y_n))
        y = np.linspace(0, y_n, y_n)
        for i in np.arange(x_n):
            u[0][i, :] = eps*np.sin(2*np.pi*y/y_n)

    f = equilibrium(rho, u)  # generate the equilibrium distribution

    # main for loop that runs based on the passed time steps
    for t in range(time_steps):
        # apply a stream operation to the distribution
        stream(f)
        # generate the new density and velocity via the collide function
        rho, u = collide(f, omega)

        if sys.argv[1] == 'shear_wave_decay_density':
            # if it's density shear wave decay add the maximum current variation / swing
            # from the initial density as a point to be analyzed
            points.append(np.max(np.abs(rho - rho_0)))
        else:
            # if it's velocity shear wave decay add the maximum current velocity as a point to be analyzed
            points.append(np.max(np.abs(u[0])))

    if sys.argv[1] == 'shear_wave_decay_density':
        # for density decay we analyze all the X direction points
        size = x_n
        # creates an numpy array of all the points we analyzed during the time steps
        stokes_points = np.array(points)
        # we only take the maximum of the swinging via the argrelextrema function
        x = argrelextrema(stokes_points, np.greater)
        stokes_points = stokes_points[x]
        # squeez removes axes and returns 1D array
        x = np.array(x).squeeze()

    else:
        # for velocity decay we analyze all the Y direction points
        size = y_n
        # creates an numpy array of all the points we analyzed during the time steps
        stokes_points = np.array(points)
        # evenly spaced values from 0 to time_steps
        x = np.arange(0, time_steps)

    # curve_fit is used to approximate the optimal viscosity ν from the observations
    # moment fluctuations decay exponentially and such a decay is represented in the lambda function
    value = curve_fit(lambda t, v: eps * np.exp(-v *
                      (2 * np.pi / size) ** 2 * t), x, stokes_points)[0][0]
    # calculated viscosity is added to the simulated array
    simulated_viscosity.append(value)

    # the viscosity is calculated via the analytic formual given in milestone 2 and added to the analytical array
    analytical_viscosity.append(1/3 * (1/omega - 0.5))


def rigidWall(wall_local):
    # the rigidWall function applies the bounce back boundary condition
    # depending on the argument we pass when calling this function that specifies where we want to apply this
    # we transform different segments of the distribution array
    if wall_local == 'top':
        # roll back for channels 2, 5 and 6 to bounce to channels 4, 7 and 8 respectivaly
        f[[4, 7, 8], :, -1] = f[[2, 5, 6], :, -1]
    elif wall_local == 'bottom':
        # roll back for channels 4, 7 and 8 to bounce to channels 2, 5 and 6 respectivaly
        f[[2, 5, 6], :, 0] = f[[4, 7, 8], :, 0]
    elif wall_local == 'right':
        # roll back for channels 1, 5 and 8 to bounce to channels 3, 7 and 6 respectivaly
        f[[3, 7, 6], -1, :] = f[[1, 5, 8], -1, :]
    else:
        # roll back for channels 3, 7 and 6 to bounce to channels 1, 5 and 8 respectivaly
        f[[1, 5, 8], 0, :] = f[[3, 7, 6], 0, :]


def movingWall(wall_local):
    # the movingWall function applies the bounce back boundary condition with momentum
    avg_rho = np.mean(rho)
    # we precompute all the scalars together which will be used for the momentum calculation
    # it will be applied to all the velocity dimensions based on their 'weight' (w) and the average density in that moment
    multiplier = 2 * avg_rho * w / c_s**2

    # based on the wall we decide to shift the wall velocity array will have as a wall velocity
    # either the first or second element and the other element will be 0
    # we apply the boundary condition on different parts of the distribution array based on the argument we pass the function
    if wall_local == 'top':
        wall_velocity_local = [wall_velocity, 0.0]
        # roll back for channel 2 to bounce (with momentum) to channel 4
        f[4, :, -1] = f[2, :, -1] - multiplier[2] * c[2] @ wall_velocity_local
        # roll back for channel 5 to bounce (with momentum) to channel 7
        f[7, :, -1] = f[5, :, -1] - multiplier[5] * c[5] @ wall_velocity_local
        # roll back for channel 6 to bounce (with momentum) to channel 8
        f[8, :, -1] = f[6, :, -1] - multiplier[6] * c[6] @ wall_velocity_local
    elif wall_local == 'bottom':
        wall_velocity_local = [wall_velocity, 0.0]
        # roll back for channel 4 to bounce (with momentum) to channel 2
        f[2, :, 0] = f[4, :, 0] - multiplier[4] * c[4] @ wall_velocity_local
        # roll back for channel 7 to bounce (with momentum) to channel 5
        f[5, :, 0] = f[7, :, 0] - multiplier[7] * c[7] @ wall_velocity_local
        # roll back for channel 8 to bounce (with momentum) to channel 6
        f[6, :, 0] = f[8, :, 0] - multiplier[8] * c[8] @ wall_velocity_local
    elif wall_local == 'right':
        wall_velocity_local = [0.0, wall_velocity]
        # roll back for channel 1 to bounce (with momentum) to channel 3
        f[3, -1] = f[1, -1] - multiplier[1] * c[1] @ wall_velocity_local
        # roll back for channel 5 to bounce (with momentum) to channel 7
        f[7, -1] = f[5, -1] - multiplier[5] * c[5] @ wall_velocity_local
        # roll back for channel 8 to bounce (with momentum) to channel 6
        f[6, -1] = f[8, -1] - multiplier[8] * c[8] @ wall_velocity_local
    else:
        wall_velocity_local = [0.0, wall_velocity]
        # roll back for channel 3 to bounce (with momentum) to channel 1
        f[1, 0] = f[3, 0] - multiplier[3] * c[3] @ wall_velocity_local
        # roll back for channel 7 to bounce (with momentum) to channel 5
        f[5, 0] = f[7, 0] - multiplier[7] * c[7] @ wall_velocity_local
        # roll back for channel 6 to bounce (with momentum) to channel 8
        f[8, 0] = f[6, 0] - multiplier[6] * c[6] @ wall_velocity_local


def flow():
    # we generate the inlet and outlet flow arrays based on the grid size on the x dimention
    # we added the c_s**2 scalar due to the ideal gas equation of state
    rho_in = np.ones((x_n)) * inlet_rho * c_s**2
    rho_out = np.ones((x_n)) * outlet_rho * c_s**2

    # based on the direction of flow we pass as argument we shift different parts of the distribution array
    # it calculates the equilibrium distribution as well as that of the inlet and outlet density
    # with the velocity at inlet/outlet position
    if flow_direction == 'right':
        f_in = equilibrium(rho_in, u[:, -1, :])
        f_out = equilibrium(rho_out, u[:, 0, :])
        f_eq = equilibrium(rho, u)

        # the distribution at the inlet postion is calculated as the inlet flow distribution with the difference
        # between the normal and equilibrium distributions at outlet postition
        inlet = f_in + (f[:, -1] - f_eq[:, -1])
        # the distribution at the outlet position is calculated as the outlet flow distribution with the difference
        # between the normal and equilibrium distributions at inlet postition
        outlet = f_out + (f[:, 0] - f_eq[:, 0])

        # only the affected channels 1, 5 and 8 are assigned the computed distribution at inlet
        f[[1, 5, 8], 0] = inlet[[1, 5, 8]]
        # only the affected channels 3, 6 and 7 are assigned the computed distribution at outlet
        f[[3, 6, 7], -1] = outlet[[3, 6, 7]]

    else:
        f_in = equilibrium(rho_in, u[:, 0, :])
        f_out = equilibrium(rho_out, u[:, -1, :])
        f_eq = equilibrium(rho, u)

        # the distribution at the inlet postion is calculated as the inlet flow distribution with the difference
        # between the normal and equilibrium distributions at outlet postition
        inlet = f_in + (f[:, 0] - f_eq[:, 0])
        # the distribution at the outlet position is calculated as the outlet flow distribution with the difference
        # between the normal and equilibrium distributions at inlet postition
        outlet = f_out + (f[:, -1] - f_eq[:, -1])

        # only the affected channels 3, 6and 7 are assigned the computed distribution at inlet
        f[[3, 6, 7], -1] = inlet[[3, 6, 7]]
        # only the affected channels 1, 5 and 8 are assigned the computed distribution at outlet
        f[[1, 5, 8], 0] = outlet[[1, 5, 8]]


if sys.argv[1] == "shear_wave_decay_density":
    u = np.zeros((2, x_n, y_n))  # initialize velocity to 0
    rho = np.zeros((x_n, y_n))  # initialize density to 1
    # returns the even spaced numbers from 0 to X grid size
    x = np.linspace(0, x_n, x_n)

    # set a sinisuodal as initial density to see how it will decay
    for i in np.arange(y_n):
        rho[:, i] = rho_0 + eps*np.sin(2*np.pi*x/x_n)

    # generate the equilibrium distribution
    f = equilibrium(rho, u)

    avg_rho = []
    # create directory to save the simulations that will be generated
    os.makedirs('./shear_wave_decay_density', exist_ok=True)

    # main for loop that runs based on the passed time steps
    for t in range(time_steps):
        # apply a stream operation to the distribution
        stream(f)
        # generate the new density and velocity via the collide function
        rho, u = collide(f, omega)
        # add the current average density into the average density array for later computations
        avg_rho.append(rho[int(x_n/4), int(y_n/2)])

        # once evry 150 steps plot the current density distribution at the center of the grid
        if t % 150 == 0 and t <= 1500:
            plt.clf()
            plt.ylim([rho_0 - eps, rho_0 + eps])
            plt.ylabel('density rho at y = '+str(int(y_n/2)))
            plt.xlabel('x position')
            plt.title('t = '+str(t))
            plt.plot(rho[:, int(y_n/2)])
            plt.savefig('shear_wave_decay_density/t=' +
                        str(t)+'.png', bbox_inches='tight')

    # once the specified time steps have finished we plot the average density distribution in order
    # to see how it evolved in time
    plt.clf()
    plt.ylim([rho_0 - eps, rho_0 + eps])
    plt.ylabel('density rho at x = '+str(int(x_n/2))+' y = '+str(int(y_n/2)))
    plt.xlabel('timestep t')
    plt.plot(avg_rho)
    plt.savefig('shear_wave_decay_density/evolution_over_time.png',
                bbox_inches='tight')

    # apply the stokes condition for the specified range of omegas
    # the results are saved on the analytical and simulated viscosity variables
    for x in W:
        stokes_condition(x)

    # plot the generated simulations using a log scale for the viscosity
    plt.clf()
    plt.yscale('log')
    plt.ylabel('viscosity v')
    plt.xlabel('relaxation term w')
    plt.plot(W, analytical_viscosity, color='orange',
             label='analytic viscosity')
    plt.plot(W, simulated_viscosity, color='cornflowerblue',
             label='simulated viscosity')
    plt.legend()
    plt.savefig('shear_wave_decay_density/w_relaxation.png',
                bbox_inches='tight')


elif sys.argv[1] == "shear_wave_decay_velocity":
    rho = np.ones((x_n, y_n))  # initialize density to 1
    u = np.zeros((2, x_n, y_n))  # initialize velocity to 0
    # returns the even spaced numbers from 0 to Y grid size
    y = np.linspace(0, y_n, y_n)

    # set a sinisuodal as initial veolcity to see how it will decay
    for i in np.arange(x_n):
        u[0][i, :] = eps*np.sin(2*np.pi*y/y_n)

    # generate the equilibrium distribution
    f = equilibrium(rho, u)

    avg_u = []
    # create directory to save the simulations that will be generated
    os.makedirs('./shear_wave_decay_velocity', exist_ok=True)

    # main for loop that runs based on the passed time steps
    for t in range(time_steps):
        # apply a stream operation to the distribution
        stream(f)
        # generate the new density and velocity via the collide function
        rho, u = collide(f, omega)
        # add the current average velocity into the average velocity array for later computations
        avg_u.append(u[0, int(x_n/2), int(y_n/4)])

        # once evry 150 steps plot the current density distribution at the center of the grid
        if t % 150 == 0 and t <= 1500:
            plt.clf()
            plt.ylim([-eps, +eps])
            plt.ylabel('velocity u at x = '+str(int(x_n/2)))
            plt.xlabel('y position')
            plt.title('t = '+str(t))
            plt.plot(u[0, int(x_n/2)])
            plt.savefig('shear_wave_decay_velocity/t=' +
                        str(t)+'.png', bbox_inches='tight')

    # once the specified time steps have finished we plot the average veolcity distribution in order
    # to see how it evolved in time
    plt.clf()
    plt.ylim([0, eps])
    plt.ylabel('velocity u at x = '+str(int(x_n/2))+' y = '+str(int(y_n/4)))
    plt.xlabel('timestep t')
    plt.plot(avg_u)
    plt.savefig('shear_wave_decay_velocity/evolution_over_time.png',
                bbox_inches='tight')

    # apply the stokes condition for the specified range of omegas
    # the results are saved on the analytical and simulated viscosity variables
    for x in W:
        stokes_condition(x)

    # plot the generated simulations using a log scale for the viscosity
    plt.clf()
    plt.yscale('log')
    plt.ylabel('viscosity v')
    plt.xlabel('relaxation term w')
    plt.plot(W, analytical_viscosity, color='orange',
             label="analytic viscosity")
    plt.plot(W, simulated_viscosity, color='cornflowerblue',
             label="simulated viscosity")
    plt.legend()
    plt.savefig('shear_wave_decay_velocity/w_relaxation.png')


elif sys.argv[1] == "couette_flow":
    rho = np.ones((x_n, y_n))  # initialize density to 1
    u = np.zeros((2, x_n, y_n))  # initialize velocity to 0
    f = equilibrium(rho, u)  # generate the equilibrium distribution

    y = np.arange(y_n)  # evenly spaced values from 0 to Y

    # if we specify the top wall as moving wall then the bottom wall is defined as rigid wall and vice versa
    # the analytical velocity is taken form the 'A Graphical Technique for Solving the Couette-Poiseuille
    # Problem for Generalized Newtonian Fluids' paper
    # depending on the moving wall the analytical velocity also changes orientation
    if wall == 'top':
        rigid = 'bottom'
        analytical_viscosity = (y) / (y_n-1) * wall_velocity
    else:
        rigid = 'top'
        analytical_viscosity = (y_n-1 - y) / (y_n-1) * wall_velocity

    # create directory to save the simulations that will be generated
    os.makedirs('./couette_flow', exist_ok=True)

    # define plot attributes
    plt.ylim([-1, y_n])
    plt.ylabel('y position')
    plt.xlabel('velocity u at x = '+str(int(x_n/2)))

    # main for loop that runs based on the passed time steps
    for t in range(time_steps):
        # apply a stream operation to the distribution
        stream(f)
        # apply moving wall boundary conditions to upper/lower wall
        movingWall(wall)
        # apply rigid wall boundary conditions to lowwer/upper wall
        rigidWall(rigid)
        # generate the new density and velocity via the collide function
        rho, u = collide(f, omega)

        # add the calculation to the plot every 200 steps (include the label only once)
        if t == 0:
            plt.plot(u[0, int(x_n/2)], y, color='cornflowerblue',
                     label='Simulated viscosity')
        elif t % 200 == 0:
            plt.plot(u[0, int(x_n/2)], y, color='cornflowerblue')

    # add the analytical velocity to the plot
    plt.plot(analytical_viscosity, y, color='orange',
             label='Analytical viscosity')

    # add the rigid wall to the plot
    plt.axhline(y=-0.5 if wall == 'top' else y_n-0.5,
                color='black', label='Rigid Wall')
    # add the moving wall to the plot
    plt.axhline(y=y_n-0.5 if wall == 'top' else -
                0.5, color='red', label='Moving Wall')

    plt.legend()
    plt.savefig('couette_flow/couette_flow_'+wall+'_t=' +
                str(time_steps)+'.png', bbox_inches='tight')


elif sys.argv[1] == "poiseuille_flow":
    rho = np.ones((x_n, y_n))  # initialize density to 1
    u = np.zeros((2, x_n, y_n))  # initialize velocity to 0
    f = equilibrium(rho, u)  # generate the equilibrium distribution

    # create directory to save the simulations that will be generated
    os.makedirs('./poiseuille_flow', exist_ok=True)

    plt.ylim([-1, y_n])
    plt.ylabel('y position')
    plt.xlabel('velocity u at x = '+str(int(x_n/2)))

    y = np.arange(y_n)

    # viscosity formula is given on milestone 2
    v = 1/3 * (1/omega - 0.5)

    # the analytical flow is taken form the 'A Graphical Technique for Solving the Couette-Poiseuille
    # Problem for Generalized Newtonian Fluids' paper
    derivative = c_s**2 * delta_rho / (x_n)
    avg_rho = np.mean(rho)
    if flow_direction == 'right':
        analytical_viscsity = -1/(2*avg_rho*v)*derivative*y*(y_n-1-y)
    else:
        analytical_viscsity = 1/(2*avg_rho*v)*derivative*y*(y_n-1-y)

    # main for loop that runs based on the passed time steps
    for t in range(time_steps):
        # apply the flow function
        flow()
        # apply the streaming function to the distribution array
        stream(f)
        # apply the rigid wall boundary conditions to the upper wall
        rigidWall('top')
        # apply the rigid wall boundary conditions to the lower wall
        rigidWall('bottom')
        # generate the new density and velocity via the collide function
        rho, u = collide(f, omega)

        # add the calculation to the plot every 200 steps (include the label only once)
        if t == 0:
            plt.plot(u[0, int(x_n/2)], y, color='cornflowerblue',
                     label='Simulated flow profile')
        elif t % 500 == 0:
            plt.plot(u[0, int(x_n/2)], y, color='cornflowerblue')

    # plot the analytical flow profile
    plt.plot(analytical_viscsity, y, color='orange',
             label='Analytical flow profile')

    # plot the upper and lower rigid walls
    plt.axhline(y=-0.5, color='black', label='Rigid Wall')
    plt.axhline(y=y_n-0.5, color='black')

    plt.legend()
    plt.savefig('poiseuille_flow/poiseuille_flow_'+str(flow_direction)+'_t=' +
                str(time_steps)+'.png', bbox_inches='tight')


elif sys.argv[1] == "sliding_lid":
    rho = np.ones((x_n, y_n))  # initialize density to 1
    u = np.zeros((2, x_n, y_n))  # initialize velocity to 0
    f = equilibrium(rho, u)  # generate the equilibrium distribution

    # in the sliding lid experiment viscosity is calculated based on the wall velocity and the Reynolds
    # with the formula given in milestone 6
    v = x_n * wall_velocity / Re

    # we flip the viscosity formula to now generate the relaxation parameter
    omega = 1 / (0.5 + 3 * v)

    # we check that omega does not go above 1.7 since it leads to inacurrecy in calculations
    if omega > 1.7:
        print('omega greater than 1.7')
        exit()

    x = np.arange(x_n)
    y = np.arange(y_n)

    # create directory to save the simulations that will be generated
    os.makedirs('./sliding_lid_Re='+str(Re), exist_ok=True)

    # main for loop that runs based on the passed time steps
    for t in range(time_steps+1):
        # apply a stream operation to the distribution
        stream(f)
        # apply moving wall boundary conditions to upper wall
        movingWall('top')
        # apply the rigid wall boundary conditions to the right wall
        rigidWall('right')
        # apply the rigid wall boundary conditions to the left wall
        rigidWall('left')
        # apply the rigid wall boundary conditions to the lower wall
        rigidWall('bottom')
        # generate the new density and velocity via the collide function
        rho, u = collide(f, omega)

        # plot the stream every 500 steps
        if t % 500 == 0:
            plt.clf()
            plt.ylim([-1, y_n])
            plt.xlim([-1, x_n])
            plt.ylabel('y position')
            plt.xlabel('x position')
            plt.streamplot(x, y, u[0].T, u[1].T,
                           density=1, color='cornflowerblue')
            plt.savefig('sliding_lid_Re='+str(Re)+'/sliding_lid_t=' +
                        str(t)+'.png', bbox_inches='tight')


elif sys.argv[1] == "scaling_plot":
    x = [4, 9, 16, 25, 36, 100, 144]  # number of processors

    # MLUPS for each nnumber of processes
    # MLUPS = the number of grid points * number of time steps / runtime
    # time steps = 100000
    line1 = [85, 66, 56, 62, 57, 50, 47]
    line2 = [633, 322, 208, 168, 139, 87, 75]
    line3 = [9999, 1045, 598, 459, 355, 141, 112]

    # plot the scale test in log scale
    plt.yscale('log')
    plt.plot(x, line1, label='100x100')
    plt.plot(x, line2, label='300x300')
    plt.plot(x, line3, label='500x500')
    plt.legend(loc='best')
    plt.savefig('scaling_plot', bbox_inches='tight')

else:
    print('Invalid Argument! Please use one of the proper arguments below:')
    print('- shear_wave_decay_density')
    print('- shear_wave_decay_velocity')
    print('- couette_flow')
    print('- poiseuille_flow')
    print('- sliding_lid')
    print('- scaling_plot')

    print('Optional Arguments include:')
    print('-xn (int) : Grid Size on X dimension')
    print('-yn (int) : Grid Size on Y dimension')
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
    print(' -r (int) : Reynolds number')
