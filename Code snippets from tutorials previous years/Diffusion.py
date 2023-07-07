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

# # Diffusion equation
#
# $$\frac{\partial\phi(x,t)}{\partial t} = D\frac{\partial^2\phi(x,t)}{\partial x^2}$$
# while its discretized version reads
# $$\phi(x_j,t+\Delta t) = \phi(x_j,t)+\frac{D\Delta t}{\Delta x^2}
# \left(\phi(x_{j+1},t)-2\phi(x_j,t)+\phi(x_{j-1},t)\right)$$
# For choosing $\Delta t$ please not that
# $$\Delta t\le 0.5 \frac{\Delta x^2}{D}$$

import numpy as np
import matplotlib.pyplot as plt
import time

# Initialize the field phi
D = 1.0
dx = 0.1
dt = 0.001
nx = 1000
x = np.arange(nx)*dx
#
phi = np.zeros(nx)
sigma0 = 20*dx
phi = np.exp(-(x-nx*dx/2)**2/(2*sigma0**2)) / (np.sqrt(2*np.pi)*sigma0)
phi0 = phi

# Do some iterations and observe what happens. Measure execution time
n = 200000
start_time = time.time()
for i in np.arange(n):
    phi = phi + D*dt*(np.roll(phi,1)-2.*phi+np.roll(phi,-1))/dx**2
end_time = time.time()
print("Execution time for {} timesteps was {}s".format(n, end_time - start_time))

# %matplotlib notebook
plt.plot(phi)
plt.plot(phi1)
plt.plot(phi0)

phi1 = phi


