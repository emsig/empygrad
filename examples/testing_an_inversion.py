import matplotlib.pyplot as plt
import numpy as np
from discretize import TensorMesh
from wbi import wavelet_regularization as regularization
from scipy.optimize import minimize

##

from empygrad.model import dipole

## Defining the Model and Mapping

# Here we generate a synthetic model and a mappig which goes from the model
# space to the row space of our linear operator.

nParam = 50  # Number of model parameters
fact = 0.1 # factor to stretch the model in depth
depth = np.linspace(0,20, nParam)*fact
# A 1D mesh
mesh = TensorMesh([np.r_[np.diff(depth),1]])

# Creating the true model
true_model = np.ones(nParam)*0.1
true_model[depth > 6*fact] = 0.15
true_model[depth > 11*fact] = 0.35
true_model[depth > 15*fact] = 0.1

# Plotting the true model
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)
ax.plot(depth, true_model, "b-")
ax.set_ylim([0, 1])
plt.show()

## Defining the forward model


EC = np.r_[1e-8, true_model]  # Adding the air-layer
s = np.arange(1,40)
freq = 100
d = depth

inpdat = {'src': [0, 0, 0.1, ], 'rec': [s, np.zeros(s.shape), 0.1, ],
          'depth': d, 'freqtime': freq, 'aniso': np.ones(EC.size),
        'verb': 1,'ab': 11}

F, J = dipole(**inpdat, res=1/np.r_[1e-8, true_model], jac='res')

plt.plot(F.imag, "ro-")
plt.show()

plt.figure()
plt.plot(depth, J[0,0, 1:].imag, "ko-")
ax = plt.gca()
ax1 = ax.twinx()
ax1.plot(depth, true_model, "b-")
plt.show()

## Predict Synthetic Data

# Here, we use the true model to create synthetic data which we will subsequently
# invert.

# Standard deviation of Gaussian noise being added
std = 0.01
np.random.seed(42)
dclean = F.imag
dobs = dclean
nd = dobs.size
W = 1/(std*dobs) # reciprocals of estimated noise


# Define the Inverse Problem
# The inverse problem is defined by 3 things:

#     1) Data Misfit: a measure of how well our recovered model explains the field data
#     2) Regularization: constraints placed on the recovered model and a priori information
#     3) Optimization: the numerical approach used to solve the inverse problem

# Define the data misfit. Here the data misfit is the L2 norm of the weighted
# residual between the observed data and the data predicted for a given model.
# Within the data misfit, the residual between predicted and observed data are
# normalized by the data's standard deviation.

def dmisfit(m):
    res = 1/np.r_[1e-8, m]
    F = dipole(**inpdat, res=res).imag
    return 1 / nd * np.linalg.norm(W * (F - dobs)) ** 2

def dmisfit_deriv(m):
    res = 1/np.r_[1e-8, m]
    F, J = dipole(**inpdat, res=res, jac='res')
    F = F.imag
    J = J[0,:, :].imag
    #Removing the air-layer
    J = J[:, 1:]
    deriv = J.T @ (W**2 * (F - dobs))
    # chain rule for res = 1/m -> d(res)/d(m) = -1/m^2
    deriv = -deriv / (m**2)
    return 2 / nd * deriv

##
from scipy.optimize import check_grad
"""
def wrap_F(m):
    return dipole(**inpdat, res=1/np.r_[1e-8, m], jac='res')[0].imag

def wrap_J(m):
    return dipole(**inpdat, res=1/np.r_[1e-8, m], jac='res')[1].imag[0, :, 1:]

print(check_grad(wrap_F, wrap_J, np.random.rand(nParam)))"""

print("Gradient check: ", check_grad(dmisfit, dmisfit_deriv, np.ones(nParam)))
##
# Define the regularization (model objective function).

# Play here with the wav-parameter
# - db1 = blocky
# - db2, db3, db4 = rather sharp
# - db5+ = rather smooth

reg = regularization.WaveletRegularization1D(mesh, wav="db1")
beta = 1e3

def phi(m):
    m = np.exp(m) # We work in log-domain to ensure positive EC-values
    return dmisfit(m) + beta * reg(m)


def jac(m):
    m = np.exp(m) # We work in log-domain to ensure positive EC-values
    deriv = dmisfit_deriv(m) + beta * reg.deriv(m)
    return deriv * np.exp(m)

starting_model = np.ones(nParam)*np.log(0.2) # Note, we work in log-domain to ensure positive EC-values
x = minimize(phi, starting_model, jac=jac, method='L-BFGS-B', options={'maxiter':200} )

m = x.x

## Plotting Results

# Observed versus predicted data
fig, ax = plt.subplots(1, 2, figsize=(12 * 1.2, 4 * 1.2))
ax[0].semilogy(-dobs, "b-")
ax[0].plot(-dipole(**inpdat, res=1/np.r_[1e-8, np.exp(starting_model)]).imag, "k-")
ax[0].plot(-dipole(**inpdat, res=1/np.r_[1e-8, np.exp(x.x)]).imag, "r-")
ax[0].legend(("Observed Data","Starting Model", "Predicted Data"))

# True versus recovered model
ax[1].plot(mesh.cell_centers_x, true_model, "b-")
ax[1].plot(mesh.cell_centers_x, np.exp(starting_model), "k-")
ax[1].plot(mesh.cell_centers_x, np.exp(x.x), "r-")
ax[1].legend(("True Model", "Starting Model","Recovered Model"))
# ax[1].set_ylim([-2, 2])
ax[1].set_title("Wavelet-type " + reg.wavelets.wav)
plt.show()

print('True model: ', dmisfit(true_model))
print('Starting model: ', dmisfit(np.exp(starting_model)))
print('Recovered model: ', dmisfit(np.exp(x.x)))

##
fig, ax_ls = plt.subplots(2,2)
wav_list = ['db1', 'db2', 'db3', 'db6']
betalist = [1e2, 5e1, 5e1, 1e4]
for idx, wav in enumerate(wav_list):
    reg = regularization.WaveletRegularization1D(mesh, wav=wav)
    beta = betalist[idx]
    def phi(m):
        return dmisfit(m) + beta * reg(np.exp(m))


    def jac(m):
        deriv = dmisfit_deriv(m) + beta * reg.deriv(np.exp(m))
        return deriv * np.exp(m)

    starting_model = np.random.rand(nParam) * np.log(0.1) # Note, we work in log-domain to ensure positive EC-values
    x = minimize(phi, starting_model, jac=jac, method='L-BFGS-B', options={'maxiter':250} )
    ax_ls[idx//2, idx%2].plot(mesh.cell_centers_x, true_model, "b-")
    ax_ls[idx//2, idx%2].plot(mesh.cell_centers_x, np.exp(x.x), "r-")
    ax_ls[idx//2, idx%2].legend(("True Model", "Recovered Model"))
    # ax[1].set_ylim([-2, 2])
    ax_ls[idx//2, idx%2].set_title("Wavelet-type " + reg.wavelets.wav)
plt.tight_layout()
plt.show()
##
