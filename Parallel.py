import sys
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

s = 4
x_n = y_n = 60
time_steps = 3000
omega = 1
wall_velocity = 0.1


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
        case '-v':
            wall_velocity = float(sys.argv[s-1])
    s += 2

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
    if x_n < y_n:
        sectsX = int(np.floor(np.sqrt(size*x_n/y_n)))
        sectsY = int(np.floor(size/sectsX))
        print(
            'We have {} fields in x-direction and {} in y-direction'.format(sectsX, sectsY))
        print('How do the fractions look like?')
        print('x_n/y_n={} and sectsX/sectsY = {}\n'.format(x_n/y_n, sectsX/sectsY))
    elif x_n > y_n:
        sectsY = int(np.floor(np.sqrt(size*y_n/x_n)))
        sectsX = int(np.floor(size/sectsY))
        print(
            'We have {} fields in x-direction and {} in y-direction'.format(sectsX, sectsY))
        print('How do the fractions look like?')
        print('x_n/y_n={} and sectsX/sectsY = {}\n'.format(x_n/y_n, sectsX/sectsY))
    elif x_n == y_n:
        sectsY = int(np.floor(np.sqrt(size)))
        sectsX = int(size/sectsY)
        if rank == 0:
            print('In the case of equal size we divide the processes as {} and {}'.format(
                sectsX, sectsY))

    return (sectsX, sectsY)


def x_in_process(x_coord):
    lower = rank_coords[0] * (x_n // sectsX)
    upper = (rank_coords[0] + 1) * (x_n // sectsX) - \
        1 if not rank_coords[0] == sectsX - 1 else x_n - 1
    return lower <= x_coord <= upper


def y_in_process(y_coord):
    lower = rank_coords[1] * (y_n // sectsY)
    upper = (rank_coords[1] + 1) * (y_n // sectsY) - \
        1 if not rank_coords[1] == sectsY - 1 else y_n - 1
    return lower <= y_coord <= upper


def global_to_local_direction(coord1d, global_dir, lattice_dir, dir_size):
    # +1 due to ghost cell
    return int(global_dir - coord1d * (lattice_dir // dir_size)) + 1


def global_coord_to_local_coord(global_x, global_y):
    if x_in_process(global_x) and y_in_process(global_y):
        local_x = global_to_local_direction(
            rank_coords[0], global_x, x_n, sectsX)
        local_y = global_to_local_direction(
            rank_coords[1], global_y, y_n, sectsY)
        return rank_coords, local_x, local_y
    return None, None, None


def Communicate(f, cartcomm, sd):
    sR, dR, sL, dL, sU, dU, sD, dD = sd
    rby = np.zeros((9, f.shape[-1]))
    rbx = np.zeros((9, f.shape[-2]))

    sb = f[:, :, 1].copy()
    cartcomm.Sendrecv(sb, dL, recvbuf=rbx, source=sL)
    f[:, :, -1] = rbx

    sb = f[:, :, -2].copy()
    cartcomm.Sendrecv(sb, dR, recvbuf=rbx, source=sR)
    f[:, :, 0] = rbx

    sb = f[:, 1, :].copy()
    cartcomm.Sendrecv(sb, dU, recvbuf=rby, source=sU)
    f[:, -1, :] = rby

    sb = f[:, -2, :].copy()
    cartcomm.Sendrecv(sb, dD, recvbuf=rby, source=sD)
    f[:, 0, :] = rby


def parallel_boundary_conditions():

    if y_in_process(y_n-1):
        # Moving top wall
        avg_rho = np.mean(rho)
        multiplier = 2 * avg_rho * w / c_s**2
        wall_velocity_local = [wall_velocity, 0.0]

        f[4, :, -1] = f[2, :, -1] - \
            multiplier[2] * c[2] @ wall_velocity_local
        f[7, :, -1] = f[5, :, -1] - \
            multiplier[5] * c[5] @ wall_velocity_local
        f[8, :, -1] = f[6, :, -1] - \
            multiplier[6] * c[6] @ wall_velocity_local

    if y_in_process(0):
        # Rigid bottom wall
        f[[2, 5, 6], :, 0] = f[[4, 7, 8], :, 0]

    if x_in_process(x_n-1):
        # Rigid right wall
        f[[3, 7, 6], -1, :] = f[[1, 5, 8], -1, :]

    if x_in_process(0):
        # Rigid left wall
        f[[1, 5, 8], 0, :] = f[[3, 7, 6], 0, :]


comm = MPI.COMM_WORLD
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()

sectsX, sectsY = getSections()

local_x_n = int(x_n // sectsX)
local_y_n = int(y_n // sectsY)

boundary = [False, False, False, False]
cartcomm = comm.Create_cart(dims=[sectsX, sectsY], periods=[
                            True, True], reorder=False)
rank_coords = cartcomm.Get_coords(rank)

# where to receive from and where send to
sR, dR = cartcomm.Shift(1, 1)
sL, dL = cartcomm.Shift(1, -1)
sU, dU = cartcomm.Shift(0, -1)
sD, dD = cartcomm.Shift(0, 1)

sd = np.array([sR, dR, sL, dL, sU, dU, sD, dD], dtype=int)
allrank_coords = comm.gather(rank_coords, root=0)
allDestSourBuf = np.zeros(size*8, dtype=int)
comm.Gather(sd, allDestSourBuf, root=0)

if rank == 0:
    cartarray = np.ones((sectsY, sectsX), dtype=int)
    allDestSour = np.array(allDestSourBuf).reshape((size, 8))

    for i in np.arange(size):
        cartarray[allrank_coords[i][0], allrank_coords[i][1]] = i
        sR, dR, sL, dL, sU, dU, sD, dD = allDestSour[i]

rho = np.ones((local_x_n + 2, local_y_n + 2))
u = np.zeros((2, local_x_n + 2, local_y_n + 2))
f = equilibrium(rho, u)

Re = 100
v = (local_x_n+2) * wall_velocity / Re
omega = 1 / (0.5 + 3 * v)

os.makedirs('./sliding_lid_parallelized', exist_ok=True)

f_full = np.zeros((9*x_n*y_n))
comm.Gather(f[:, 1:-1, 1:-1].reshape(9*local_x_n*local_y_n), f_full, root=0)
rank_coords_x = comm.gather(rank_coords[1], root=0)
rank_coords_y = comm.gather(rank_coords[0], root=0)

if rank == 0:
    X0, Y0 = np.meshgrid(np.arange(x_n), np.arange(y_n))
    xy = np.array([rank_coords_x, rank_coords_y]).T
    f_plot = np.zeros((9, x_n, y_n))
    f_full = f_full.reshape(9, x_n * y_n)

    for i in np.arange(sectsX):
        for j in np.arange(sectsY):
            k = i*sectsX+j
            xlo = local_x_n*xy[k, 1]
            xhi = local_x_n*(xy[k, 1]+1)
            ylo = local_y_n*xy[k, 0]
            yhi = local_y_n*(xy[k, 0]+1)
            flo = k*x_n*y_n//(sectsX*sectsY)
            fhi = (k+1)*x_n*y_n//(sectsX*sectsY)
            f_plot[:, xlo:xhi, ylo:yhi] = f_full[:,
                                                 flo:fhi].reshape(9, local_x_n, local_y_n)

    rho_plot, u_plot = collide(f_plot, omega)


if rank == 0:
    start_time = time.time()

for t in range(time_steps):
    Communicate(f, cartcomm, sd)
    stream(f)
    parallel_boundary_conditions()
    rho, u = collide(f, omega)

if rank == 0:
    end_time = time.time()
    print('{} iterations took {}s'.format(
        time_steps, end_time - start_time))

plt.figure()
plt.streamplot(np.arange(local_x_n), np.arange(local_y_n), u[0, 1:-1, 1:-1].T, u[1, 1:-1, 1:-1].T,
               density=1, color='cornflowerblue')
plt.savefig(
    'sliding_lid_parallelized/sliding_lid_parallelized_'+str(rank)+'.png', bbox_inches='tight')

f_full = np.zeros((9*x_n*y_n))
comm.Gather(f[:, 1:-1, 1:-1].reshape(9*local_x_n*local_y_n), f_full, root=0)
rank_coords_x = comm.gather(rank_coords[1], root=0)
rank_coords_y = comm.gather(rank_coords[0], root=0)

if rank == 0:
    X0, Y0 = np.meshgrid(np.arange(x_n), np.arange(y_n))
    xy = np.array([rank_coords_x, rank_coords_y]).T
    f_plot = np.zeros((9, x_n, y_n))
    f_full = f_full.reshape(9, x_n * y_n)

    for i in np.arange(sectsX):
        for j in np.arange(sectsY):
            k = i*sectsX+j
            xlo = local_x_n*xy[k, 1]
            xhi = local_x_n*(xy[k, 1]+1)
            ylo = local_y_n*xy[k, 0]
            yhi = local_y_n*(xy[k, 0]+1)
            flo = k*x_n*y_n//(sectsX*sectsY)
            fhi = (k+1)*x_n*y_n//(sectsX*sectsY)
            f_plot[:, xlo:xhi, ylo:yhi] = f_full[:, flo:fhi].reshape(9,
                                                                     local_x_n, local_y_n)

    rho_plot, u_plot = collide(f_plot, omega)

    plt.figure()
    plt.streamplot(X0, Y0, u_plot[0].T, u_plot[1].T,
                   density=1, color='cornflowerblue')
    plt.savefig(
        'sliding_lid_parallelized/sliding_lid_parallelized.png', bbox_inches='tight')
