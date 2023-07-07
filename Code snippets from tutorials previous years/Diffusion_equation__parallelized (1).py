# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # The diffusion equation - parallelized

# Note, this is only for jupyter notebooks
import ipyparallel as ipp
# Start the cluster
cluster = ipp.Cluster(engines="mpi", n=8)
client = cluster.start_and_connect_sync()
client.ids

# %px %matplotlib inline

# +
# %%px
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
#
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

print('Rank {}/{} is alive.'.format(rank, size))
# -

# #### Define the spatial grid and the time increment

# %%px
nx = 1000
dx = 0.1
nt = 10000
dt = 0.001
D = 1 # diffusion constant

# #### Domain decomposition

# +
# %%px
# Domain decomposition: set up domain boundaries
nx1 = rank*nx//size
nx2 = (rank+1)*nx//size

print('{}, Domain boundaries: {}-{}'.format(rank, nx1, nx2-1))

# We include one additional cell at the boundaries for communication purposes
x = np.arange(nx1-1, nx2+1)*dx
# -

# #### Initial condition: We use a Gaussian concentration profile initially

# +
# %%px
sigma0 = 20*dx
c = np.exp(-(x-nx*dx/2)**2/(2*sigma0**2)) / (np.sqrt(2*np.pi)*sigma0)

plt.title('rank ${}$'.format(rank))
plt.xlim(0, nx*dx)
plt.ylim(0, 0.3)
plt.plot(x, c)
# -

# #### Propagate in time

# %%px
for t in range(nt):
    # Send to right, receive from left
    comm.Sendrecv(c[-2:-1], (rank+1)%size, recvbuf=c[:1], source=(rank-1)%size)
    # Send to left, receive from right
    comm.Sendrecv(c[1:2], (rank-1)%size, recvbuf=c[-1:], source=(rank+1)%size)
    if t % (nt/5) == 0:
        plt.plot(x, c, '-', label='$t={}$'.format(t*dt))

    d2c_dx2 = (np.roll(c, 1)-2*c+np.roll(c, -1))/(dx**2)
    c += D*d2c_dx2*dt
plt.legend(loc='best')

# %%px
print(c.shape)

# #### Gather all data on rank 0 and plot

# %%px
x_full_range = np.arange(nx)*dx
c_full_range = np.zeros(nx)
comm.Gather(c[1:-1], c_full_range, root=0)
if rank == 0:
    plt.plot(x_full_range, c_full_range, '-')

# #### Gather during progation so we can plot the evolution of the concentration profile

# %%px
if rank == 0:
    x_full_range = np.arange(nx)*dx
    c_full_range = np.zeros(nx)
for t in range(nt):
    # Send to right, receive from left
    comm.Sendrecv(c[-2:-1], (rank+1)%size, recvbuf=c[:1], source=(rank-1)%size)
    # Send to left, receive from right
    comm.Sendrecv(c[1:2], (rank-1)%size, recvbuf=c[-1:], source=(rank+1)%size)
    if t % (nt/5) == 0:
        comm.Gather(c[1:-1], c_full_range, root=0)
        if rank == 0:
            plt.plot(x_full_range, c_full_range, '-', label='$t={}$'.format(t*dt))

    d2c_dx2 = (np.roll(c, 1)-2*c+np.roll(c, -1))/(dx**2)
    c += D*d2c_dx2*dt
if rank == 0:
    plt.legend(loc='best')

# %%px
if rank == 1: print(c_full_range)

# %%px
print('Process {} knows about dx = {}'.format(rank,dx))

# %%px 
c_full_range


