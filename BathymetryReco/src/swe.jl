export simulation
function simulation(param, sim_params::simulation_setup, observation::observation_data)
    solver = swe.SWESolver(sim_params.xbounds, sim_params.timestep,
                            sim_params.nx, sim_params.tinterval, g=sim_params.g,
                            kappa=sim_params.kappa, dealias=sim_params.dealias,
                            tstart=observation.tstart,
                            problemtype=sim_params.scenario, bc_file=sim_params.bc_file);
    sample_bathy = bathymetry(solver.domain.x, param)
    sim_observations, t_sim, _, _ = solver.solve(sample_bathy, sensor_pos=sim_params.sensor_pos)
    t_sim = vec(collect(0.0:sim_params.timestep:sim_params.tinterval))
    if length(t_sim) != length(observation.t)
        indices = findall(x-> x ∈ observation.t, t_sim)
        sim_observations = sim_observations[indices,:]
    end # only use sim at measured timesteps, maybe replace by proper interpolation
    return sim_observations
end

