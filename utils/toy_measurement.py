import os
import h5py
import numpy as np
from swe_solver import SWESolver

def main():
    # Get the parent directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    path = os.path.join(parent_dir, "data/toy_measurement")
    filename = "simulation_data.h5"
    full_path = os.path.join(path, filename)

    # Ensure the directory exists
    os.makedirs(path, exist_ok=True)

    # Set up the parameters for measurement simulation (# for simulation in MCMC)
    xbounds = (0., 10.)
    nx = 130 # 64
    tend = 10
    timestep = 5e-5 # 1e-3
    g = 9.81
    kappa = 0.2
    dealias = 3/2
    bathy_peak = 5

    # Create SWE solver and calculate the solution
    solver = SWESolver(xbounds, timestep, nx, tend, g, kappa, dealias)
    h_list, u_list, t_list = solver.solve(bathy_peak)

    h_array = np.array(h_list)
    u_array = np.array(u_list)
    t_array = np.array(t_list)
    solver.initial_conditions.b.change_scales(1)
    b_array = np.copy(solver.initial_conditions.b['g'])
    dx = (xbounds[1]-xbounds[0])/nx
    H_array = h_array + np.tile(b_array, (t_array.size, 1))


    with h5py.File(full_path, "w") as f:

        f.create_dataset("y", data=H_array)
        f.create_dataset("h", data=h_array)
        f.create_dataset("u", data=u_array)
        f.create_dataset("b_exact", data=b_array)
        f.create_dataset("xgrid", data=np.copy(solver.domain.x))
        f.create_dataset("t_array", data=t_array)
        f.attrs["T_N"] = tend
        f.attrs["xmin"] = xbounds[0]
        f.attrs["xmax"] = xbounds[1]
        f.attrs["dt"] = timestep
        f.attrs["dx"] = dx
        f.attrs["g"] = g
        f.attrs["k"] = kappa  # Leave this "k" in hdf5 file,
        # otherwise compute all old solutions again.
        f.attrs["M"] = nx

if __name__ == "__main__":
    main()
