import sys
import tomllib
import torch
from sbi.inference import NPE, simulate_for_sbi
from sbi import utils
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
import matplotlib.pyplot as plt
sys.path.append("./BathymetryReco/src/")

from swe_wrapper import SWESolver



with open("config.toml" , "rb") as f:
    config = tomllib.load(f)
path = config["observation"]["path"]

# Set up the parameters for measurement simulation (# for simulation in MCMC)
sim_params = config["simulation"]
xbounds = sim_params["xbounds"]
nx =  sim_params["nx"]
total_t = sim_params["tinterval"]
tstart = sim_params["tstart"]
timestep = sim_params["timestep"]
g = sim_params["g"]
kappa = sim_params["kappa"]
dealias = sim_params["dealias"]
problemtype = sim_params["scenario"]
problem_bathy = sim_params["bathymetry"]
left_bc = sim_params["bc_file"]

solver = SWESolver(xbounds, timestep, nx, total_t,
                    tstart=tstart, g=g, kappa=kappa, dealias=dealias,
                    problemtype=problemtype, bc_file=left_bc)

def gaussian_bathymetry(x, params):
    """Gaussian bathymetry function.
    Args:
        x: The x-coordinate.

    Returns:
        The bathymetry value at x.
    """
    return 0.2*torch.exp(-1/(2*params[1])*(x-params[0])**2)

def simulator(params):
    """Simulator function to be used in SBI reconstruction.

    Args:
        params (np.ndarray): Array of bathymetry parameters.

    Returns:
        np.ndarray: Simulated sensor measurements.
    """
    # Define fixed simulation parameters

    # Create SWE solver

    # Set up the bathymetry using the provided parameters
    x = torch.as_tensor(solver.domain.x)
    bathy = gaussian_bathymetry(x, params)
    # Run the simulation
    H_sensor, _, _, _ = solver.solve(bathy)

    return torch.as_tensor(H_sensor[:, 1])  # Return sensor measurements at second sensor

# Define prior over bathymetry parameters
prior_min = [3.0, 0.025]
prior_max = [5.0, 0.075]
prior = utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min),
                                    high=torch.as_tensor(prior_max))

# Check prior, simulator, consistency
prior, num_parameters, prior_returns_numpy = process_prior(prior)
simulation_wrapper = process_simulator(simulator, prior, prior_returns_numpy)
check_sbi_inputs(simulation_wrapper, prior)

theta, x = simulate_for_sbi(simulation_wrapper, proposal=prior, num_simulations=1000, num_workers=4)