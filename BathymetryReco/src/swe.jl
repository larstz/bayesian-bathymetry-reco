export simulation
function simulation(param, sim_params::SimulationSetup, observation::ObservationData)
    solver = swe.SWESolver(
        sim_params.xbounds,
        sim_params.timestep,
        sim_params.nx,
        sim_params.tinterval,
        g = sim_params.g,
        kappa = sim_params.kappa,
        dealias = sim_params.dealias,
        tstart = observation.tstart,
        problemtype = sim_params.scenario,
        bc_file = sim_params.bc_file,
    )
    return simulation(param, solver, observation)
end

"""
    simulation(param, solver::PyObject, observation::ObservationData)

Calculate the observation from the shallow water equations `solver` with the given `param`
as bathymetry, using the provided `solver` and `observation` data.
"""
function simulation(param, solver::PyObject, observation::ObservationData)
    ########################## Using Fun from ApproxFun.jl ########################
    # f = Fun(Chebyshev(solver.xbound[1]..solver.xbound[2]), param)
    # solver_bathy = f.(solver.domain.x)
    ######################## Sample bathymetry directly ########################
    if length(param) < 4 # parametrized bathymetry
        solver_bathy = bathymetry(solver.domain.x, param)
    else
        equi_x = range(solver.xbound[1], solver.xbound[2], length(param))
        sample_bathy = bathymetry(equi_x, param)
        solver_bathy = PCHIPInterpolation(sample_bathy, equi_x)(solver.domain.x)
    end
    #############################################################################

    sim_observations, t_sim, _, _ = solver.solve(solver_bathy, sensor_pos = observation.x)
    t_sim = vec(collect(0.0:solver.dt:solver.total_t))
    if length(t_sim) != length(observation.t)
        indices = findall(x -> x ∈ observation.t, t_sim)
        sim_observations = sim_observations[indices, :]
    end # only use sim at measured timesteps, maybe replace by proper interpolation
    return sim_observations
end

export swe_solver

"""
    swe_solver(sim_params::SimulationSetup)

Create a shallow water equations solver using the provided simulation parameters `sim_params`.
This function interfaces with the Python based SWEs solver.
"""
function swe_solver(sim_params::SimulationSetup)
    solver = swe.SWESolver(
        sim_params.xbounds,
        sim_params.timestep,
        sim_params.nx,
        sim_params.tinterval,
        g = sim_params.g,
        kappa = sim_params.kappa,
        dealias = sim_params.dealias,
        tstart = sim_params.tstart,
        problemtype = sim_params.scenario,
        bc_file = sim_params.bc_file,
    )
    return solver
end
