using Pkg
Pkg.activate(".")

using Turing
using PyCall

PyCall.python
# Import the SWESolver class from the utils module
@pyimport swe_wrapper as swe

xbounds = (0., 10.)
nx = 130 # 64
tend = 10
timestep = 5e-5 # 1e-3
g = 9.81
kappa = 0.2
dealias = 3/2
bathy_peak = [5, 1]

# Create SWE solver and calculate the solution
solver = swe.SWESolver(xbounds, timestep, nx, tend, g, kappa, dealias)

# Use the solver to simulate the shallow water equation with the sampled peak
simulation = solver.solve(bathy_peak)
