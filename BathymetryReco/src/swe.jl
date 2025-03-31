export simulation
function simulation(param, sim_params::simulation_setup, observation::observation_data)
    solver = swe.SWESolver(sim_params.xbounds, sim_params.timestep,
                            sim_params.nx, sim_params.tend, g=sim_params.g,
                            kappa=sim_params.kappa, dealias=sim_params.dealias,
                            tstart=observation.tstart,
                            problemtype=sim_params.scenario, bc_file=sim_params.bc_file);
    sample_bathy = bathymetry(solver.domain.x, param)
    sim_observations, _, _, _ = solver.solve(sample_bathy, sensor_pos=sim_params.sensor_pos)
    return sim_observations
end

