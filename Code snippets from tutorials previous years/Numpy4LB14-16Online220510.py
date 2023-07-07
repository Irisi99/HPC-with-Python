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

# # Numpy methods for Streaming and Collision

import numpy as np
import matplotlib.pyplot as plt

# ### The streaming operator
# For this purpose we need the numpy method roll of arrays

# Create an array and print it
a = np.arange(5*7)
print(a)

# reshape it to a rectangular array
b = a.reshape((5, 7))
print(b)

f_cij = np.ones((9, 5, 7))
c_ca = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                 [0, 0, -1, 0, 1, -1, -1, 1, 1]]).T     # These are the velocities of the channels
#

# %matplotlib notebook
plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['xtick.labeltop'] = True
# Roll it in different direction
c = np.roll(b, shift=c_ca[2], axis=(1, 0))
print(c)
#
column_labels = list('0123456')
row_labels = list('01234')
# data = np.random.rand(5, 6)
fig, ax = plt.subplots()
data = c/a[-1]
heatmap = ax.pcolor(data, cmap=plt.cm.Reds)
# put the major ticks at the middle of each cell
ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
ax.invert_yaxis()
ax.xaxis.tick_top()
#
ax.set_xticklabels(column_labels, minor=False)
ax.set_yticklabels(row_labels, minor=False)
ax.set_ylabel('axis 0, first  index')
ax.set_title('axis 1, second index')
plt.show()

for k in np.arange(9):
    f_cij[k] = np.roll(f_cij[k], c_ca[k], axis=(1, 0))

# ### Collision operator
# To calculate the outcome of the collision
# $$ f_i+\omega(f_i^{eq}-f_i)$$
# We need to know the average velocity at $\mathbf{r}$ and the denisty $n(\mathbf{r})$.

f_cij = np.ones((9, 5, 7))
c_ca = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                 [0, 0, -1, 0, 1, -1, -1, 1, 1]]).T     # These are the velocities of the channels

# equilibrium occupation numbers
w_c = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])


for c in np.arange(9):
    f_cij[c] = w_c[c]

f_cij[:] = w_c[:, np.newaxis, np.newaxis]

f_cij[1] = f_cij[1]+0.01
f_cij[3] = f_cij[3]-0.01

u_aij = np.einsum('cij,ca->aij', f_cij, c_ca)

u_aij

rho_ij = np.einsum('cij->ij', f_cij)

rho_ij
