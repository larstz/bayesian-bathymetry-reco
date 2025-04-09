import os
import tomllib

import h5py
import numpy as np
import sys
import time
sys.path.append("./BathymetryReco/src")
from swe_wrapper import SWESolver, rampFunc, gaussian_bathymetry


def main():
    """This script is used to generate the simulation data to use
    for reconstruction of bathymetry."""
    # Change to the directory of this file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    with open("config.toml" , "rb") as f:
        config = tomllib.load(f)
    path = config["observation"]["path"]

    # Ensure the directory exists
    os.makedirs(path, exist_ok=True)

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
    store = False

    # Create SWE solver and calculate the solution
    solver = SWESolver(xbounds, timestep, nx, total_t,
                        tstart=tstart, g=g, kappa=kappa, dealias=dealias,
                        problemtype=problemtype, bc_file=left_bc)
    x = solver.domain.x

    # Set up the bathymetry
    bathy_config = config["bathymetry"]
    bathy_params = bathy_config["parameters"]

    if bathy_config["parametrized"]:
        assert bathy_config["npeaks"] == len(bathy_params)//2 # Check if the number of peaks is correct
        bathy = np.zeros_like(x)
        for i in range(bathy_config["npeaks"]):
            bathy = gaussian_bathymetry(x, bathy_params[2*i:2*(i+1)])
    else:
        bathy = rampFunc(x)

    print("Start Simulation")
    start = time.time()
    H_sensor, t_array, h_array, u_array = solver.solve(bathy)
    end = time.time()
    print(f"Simulation Done after {end-start}")
    dx = (xbounds[1]-xbounds[0])/nx
    # filename
    filename = f"simulation_data_{problemtype}_{problem_bathy}_test.h5"
    full_path = os.path.join(path, filename)
    print("Saving results to: ", full_path)
    if store:
        with h5py.File(full_path, "w") as f:
            f.create_dataset("h", data=h_array)
            f.create_dataset("u", data=u_array)
            f.create_dataset("H_sensor", data=H_sensor)
            f.create_dataset("b_exact", data=bathy)
            f.create_dataset("xgrid", data=np.copy(solver.domain.x))
            f.create_dataset("t_array", data=t_array)
            f.attrs["T_N"] = total_t
            f.attrs["xmin"] = xbounds[0]
            f.attrs["xmax"] = xbounds[1]
            f.attrs["tstart"] = tstart
            f.attrs["dt"] = timestep
            f.attrs["dx"] = dx
            f.attrs["g"] = g
            f.attrs["k"] = kappa
            f.attrs["M"] = nx
            if bathy_config["parametrized"]:
                f.attrs["npeaks"] = bathy_config["npeaks"]
                f.attrs["bathy_params"] = bathy_params

if __name__ == "__main__":
    main()
