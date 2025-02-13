import os
import h5py
import numpy as np
from swe_wrapper import SWESolver, rampFunc


def main():
    """This script is used to generate the simulation data to use
    for reconstruction of bathymetry."""
    # Get the parent directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, "data/toy_measurement")


    # Ensure the directory exists
    os.makedirs(path, exist_ok=True)

    # Set up the parameters for measurement simulation (# for simulation in MCMC)
    xbounds = (1.5, 15.)
    nx =  130
    total_t = 10
    tstart = 32.
    timestep = 5e-5
    g = 9.81
    kappa = 0.2
    dealias = 3/2
    problemtype = "waterchannel"

    # Create SWE solver and calculate the solution
    solver = SWESolver(xbounds, timestep, nx, total_t,
                       tstart=tstart, g=g, kappa=kappa, dealias=dealias,
                       problemtype=problemtype)
    x = solver.domain.x
    bathy = rampFunc(x)
    print("Start Simulation")
    H_sensor, h_array, t_array, u_array = solver.solve(bathy)
    print("Simulation Done")
    dx = (xbounds[1]-xbounds[0])/nx
    # filename
    filename = f"simulation_data_{problemtype}.h5"
    full_path = os.path.join(path, filename)
    print("Saving results to: ", full_path)
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

if __name__ == "__main__":
    main()
