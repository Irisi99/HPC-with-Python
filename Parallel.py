import time
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

s = 3
x_n = y_n = 300
time_steps = 100000
Re = 1000
wall_velocity = 0.1

while len(sys.argv) >= s:
    match sys.argv[s-2]:
        case '-n':
            x_n = y_n = int(sys.argv[s-1])
        case '-w':
            wall_velocity = int(sys.argv[s-1])
        case '-r':
            Re = int(sys.argv[s-1])

v = x_n * wall_velocity / Re
omega = 1 / (0.5 + 3 * v)

c_s = (1/np.sqrt(3))

c = np.array([
    [0, 1, 0, -1, 0, 1, -1, -1, 1],
    [0, 0, 1, 0, -1, 1, 1, -1, -1]
]).T

w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
W = np.arange(0.1, 2, 0.1)


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


def getSections():
    # checks the X and Y size of the lattice grid and distributes the available processes
    # based on their raport
    if x_n < y_n:
        sectsX = int(np.floor(np.sqrt(size*x_n/y_n)))
        sectsY = int(np.floor(size/sectsX))
        print(
            'We have {} fields in x-direction and {} in y-direction'.format(sectsX, sectsY))
    elif x_n > y_n:
        sectsY = int(np.floor(np.sqrt(size*y_n/x_n)))
        sectsX = int(np.floor(size/sectsY))
        print(
            'We have {} fields in x-direction and {} in y-direction'.format(sectsX, sectsY))
    elif x_n == y_n:
        sectsY = int(np.floor(np.sqrt(size)))
        sectsX = int(size/sectsY)
        if rank == 0:
            print('In the case of equal size we divide the processes as {} and {}'.format(
                sectsX, sectsY))

    return (sectsX, sectsY)


def x_in_process(x_coord):
    # checks if global x coordinate is in the current process
    # lower x bound of current process
    lower = rank_coords[0] * (x_n // sectsX)
    upper = (rank_coords[0] + 1) * (x_n // sectsX) - \
        1 if not rank_coords[0] == sectsX - 1 else x_n - \
        1  # upper x bound of current process
    return lower <= x_coord <= upper  # checks if x is inbetween upper and lower bounds


def y_in_process(y_coord):
    # checks if global y coordinate is in the current process
    # lower y bound of current process
    lower = rank_coords[1] * (y_n // sectsY)
    upper = (rank_coords[1] + 1) * (y_n // sectsY) - \
        1 if not rank_coords[1] == sectsY - 1 else y_n - \
        1  # upper y bound of current process
    return lower <= y_coord <= upper  # checks if y is inbetween upper and lower bounds


def Communicate(f, cartcomm, sd):
    # explode the sender and recievers of the current process
    sR, dR, sL, dL, sU, dU, sD, dD = sd

    # initialize recieve buffer to hold the ghost lattice points
    rb = np.zeros((9, f.shape[-1]))

    # put lattice points on the sender buffer
    sb = f[:, :, 1].copy()
    # communicate with the process on that holds the grid points to the left
    cartcomm.Sendrecv(sb, dL, recvbuf=rb, source=sL)
    # assign recieved buffer to the ghost lattice points
    f[:, :, -1] = rb

    # put lattice points on the sender buffer
    sb = f[:, :, -2].copy()
    # communicate with the process on that holds the grid points to the right
    cartcomm.Sendrecv(sb, dR, recvbuf=rb, source=sR)
    # assign recieved buffer to the ghost lattice points
    f[:, :, 0] = rb

    # put lattice points on the sender buffer
    sb = f[:, 1, :].copy()
    # communicate with the process on that holds the grid points above
    cartcomm.Sendrecv(sb, dU, recvbuf=rb, source=sU)
    # assign recieved buffer to the ghost lattice points
    f[:, -1, :] = rb

    # put lattice points on the sender buffer
    sb = f[:, -2, :].copy()
    # communicate with the process on that holds the grid points below
    cartcomm.Sendrecv(sb, dD, recvbuf=rb, source=sD)
    # assign recieved buffer to the ghost lattice points
    f[:, 0, :] = rb

    return f


def plot(time):
    ux_full = np.zeros((x_n*y_n))
    uy_full = np.zeros((x_n*y_n))

    # gather both X and Y velocities from all processes and put them on the predifined variables
    comm.Gather(u[0, 1:-1, 1:-1].reshape(local_x_n*local_y_n), ux_full, root=0)
    comm.Gather(u[1, 1:-1, 1:-1].reshape(local_x_n*local_y_n), uy_full, root=0)

    # get the X and Y size of the process
    rank_coords_x = comm.gather(rank_coords[1], root=0)
    rank_coords_y = comm.gather(rank_coords[0], root=0)

    if rank == 0:
        X0, Y0 = np.meshgrid(np.arange(x_n), np.arange(y_n))
        xy = np.array([rank_coords_x, rank_coords_y]).T
        ux_plot = np.zeros((x_n, y_n))
        ux_full = ux_full.reshape(x_n * y_n)
        uy_plot = np.zeros((x_n, y_n))
        uy_full = uy_full.reshape(x_n * y_n)

        # go over each process and put their computed velocity into the correct part of the full velocity array
        for i in np.arange(sectsX):
            for j in np.arange(sectsY):
                k = i*sectsX+j
                xlo = local_x_n*xy[k, 1]
                xhi = local_x_n*(xy[k, 1]+1)
                ylo = local_y_n*xy[k, 0]
                yhi = local_y_n*(xy[k, 0]+1)
                ulo = k*x_n*y_n//(sectsX*sectsY)
                uhi = (k+1)*x_n*y_n//(sectsX*sectsY)
                ux_plot[xlo:xhi, ylo:yhi] = ux_full[ulo:uhi].reshape(
                    local_x_n, local_y_n)
                uy_plot[xlo:xhi, ylo:yhi] = uy_full[ulo:uhi].reshape(
                    local_x_n, local_y_n)

        plt.figure()
        plt.streamplot(X0, Y0, ux_plot.T, uy_plot.T,
                       density=1, color='cornflowerblue')
        plt.savefig(
            'sliding_lid_parallelized_Re='+str(Re)+'/sliding_lid_parallelized_grid='+str(x_n)+'_n='+str(size)+'_t='+str(time)+'.png', bbox_inches='tight')


def parallel_boundary_conditions():
    
    # we use -2 and 1 as indexes instead of 0 and -1 to take into account the ghost latice points
    # we inserted on each process' grid

    # if current process has latice points part of the top wall apply boundary condition
    if y_in_process(y_n-1):
        # Moving top wall boundary conditions
        avg_rho = np.mean(rho)
        multiplier = 2 * avg_rho * w / c_s**2
        wall_velocity_local = [wall_velocity, 0.0]

        f[4, :, -2] = f[2, :, -2] - \
            multiplier[2] * c[2] @ wall_velocity_local
        f[7, :, -2] = f[5, :, -2] - \
            multiplier[5] * c[5] @ wall_velocity_local
        f[8, :, -2] = f[6, :, -2] - \
            multiplier[6] * c[6] @ wall_velocity_local

    # if current process has latice points part of the bottom wall apply boundary condition
    if y_in_process(0):
        # Rigid bottom wall boundary conditions
        f[[2, 5, 6], :, 1] = f[[4, 7, 8], :, 1]

    # if current process has latice points part of the right wall apply boundary condition
    if x_in_process(x_n-1):
        # Rigid right wall boundary conditions
        f[[3, 7, 6], -2, :] = f[[1, 5, 8], -2, :]

    # if current process has latice points part of the left wall apply boundary condition
    if x_in_process(0):
        # Rigid left wall boundary conditions
        f[[1, 5, 8], 1, :] = f[[3, 7, 6], 1, :]


comm = MPI.COMM_WORLD  # initialize the MPI communicator
size = MPI.COMM_WORLD.Get_size()  # get the number of available processors
rank = MPI.COMM_WORLD.Get_rank()  # get the rank of the current process

sectsX, sectsY = getSections() # get the portion of processes on the X and Y direction

# detrmine the size of the local lattice of the process by dividing the lattice size
# in the X and Y dimensions by the number of processes in that direction
local_x_n = int(x_n // sectsX)
local_y_n = int(y_n // sectsY)

# initialize cartezian communicator
cartcomm = comm.Create_cart(dims=[sectsX, sectsY], periods=[
                            True, True], reorder=False)
# get the coordinates of the current process
rank_coords = cartcomm.Get_coords(rank)

# where to receive from and where send to
sR, dR = cartcomm.Shift(1, 1)
sL, dL = cartcomm.Shift(1, -1)
sU, dU = cartcomm.Shift(0, -1)
sD, dD = cartcomm.Shift(0, 1)

sd = np.array([sR, dR, sL, dL, sU, dU, sD, dD], dtype=int)
# get the coordinates of all the processes
allrank_coords = comm.gather(rank_coords, root=0)
# create the buffer 
allDestSourBuf = np.zeros(size*8, dtype=int)
comm.Gather(sd, allDestSourBuf, root=0)

if rank == 0:
    # generate a matrix where every row specifies the rank of the reciever and sender process in each direction
    cartarray = np.ones((sectsY, sectsX), dtype=int)
    allDestSour = np.array(allDestSourBuf).reshape((size, 8))
    for i in np.arange(size):
        cartarray[allrank_coords[i][0], allrank_coords[i][1]] = i
        sR, dR, sL, dL, sU, dU, sD, dD = allDestSour[i]

# add a buffer column/row on all the directions of the lattice
rho = np.ones((local_x_n + 2, local_y_n + 2)) # initialize density to 1
u = np.zeros((2, local_x_n + 2, local_y_n + 2)) # initialize velocity to 0
f = equilibrium(rho, u)

os.makedirs('./sliding_lid_parallelized_Re='+str(Re), exist_ok=True)

if rank == 0:
    start_time = time.time()

for t in range(time_steps):
    f = Communicate(f, cartcomm, sd)
    stream(f)
    parallel_boundary_conditions()
    rho, u = collide(f, omega)

if rank == 0:
    end_time = time.time()
    print('{} iterations took {}s'.format(
        time_steps, end_time - start_time))

# plot the velocity stream after all the time steps are finished
plot(time_steps)
