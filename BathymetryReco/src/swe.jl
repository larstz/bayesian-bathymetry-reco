export simulation
function simulation(param, sim_params::simulation_setup, observation::observation_data)
    solver = swe.SWESolver(sim_params.xbounds, sim_params.timestep,
                            sim_params.nx, sim_params.tinterval, g=sim_params.g,
                            kappa=sim_params.kappa, dealias=sim_params.dealias,
                            tstart=observation.tstart,
                            problemtype=sim_params.scenario, bc_file=sim_params.bc_file);
    #TODO: add possibility to pass discretization of bathymetry
    equi_x = range(sim_params.xbounds[1], sim_params.xbounds[2], length(param))
    sample_bathy = bathymetry(equi_x, param)
    solver_bathy = PCHIPInterpolation(sample_bathy, equi_x)(solver.domain.x)
    sim_observations, t_sim, _, _ = solver.solve(solver_bathy, sensor_pos=observation.x)
    t_sim = vec(collect(0.0:sim_params.timestep:sim_params.tinterval))
    if length(t_sim) != length(observation.t)
        indices = findall(x-> x ∈ observation.t, t_sim)
        sim_observations = sim_observations[indices,:]
    end # only use sim at measured timesteps, maybe replace by proper interpolation
    return sim_observations
end

function simulation(param, solver::PyObject, observation::observation_data)

    equi_x = range(solver.xbound[1], solver.xbound[2], length(param))
    sample_bathy = bathymetry(equi_x, param)
    solver_bathy = PCHIPInterpolation(sample_bathy, equi_x)(solver.domain.x)

    sim_observations, t_sim, _, _ = solver.solve(solver_bathy, sensor_pos=observation.x)
    t_sim = vec(collect(0.0:solver.dt:solver.total_t))
    if length(t_sim) != length(observation.t)
        indices = findall(x-> x ∈ observation.t, t_sim)
        sim_observations = sim_observations[indices,:]
    end # only use sim at measured timesteps, maybe replace by proper interpolation
    return sim_observations
end

export swe_solver
function swe_solver(sim_params::simulation_setup)
    solver = swe.SWESolver(sim_params.xbounds, sim_params.timestep,
                            sim_params.nx, sim_params.tinterval, g=sim_params.g,
                            kappa=sim_params.kappa, dealias=sim_params.dealias,
                            tstart=sim_params.tstart,
                            problemtype=sim_params.scenario, bc_file=sim_params.bc_file);
    return solver
end