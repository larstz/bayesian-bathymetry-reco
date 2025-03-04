using Pkg
Pkg.activate(".")
using Distributed
using TOML
using Distributions
addprocs(4)
println("Running on $(nprocs()) processes")
@everywhere using PyCall
@everywhere swe = pyimport("swe_wrapper")
using HDF5

@everywhere struct observation_data
    t::Array{Float64}
    x::Array{Float64}
    H::Array{Float64}
    tstart::Float64
end

@everywhere struct simulation_setup
    xbounds::Array{Float64, 1}
    timestep::Float64
    nx::Int
    tend::Float64
    g::Float64
    kappa::Float64
    dealias::Float64
    scenario::String
end

@everywhere function bathymetry(x::Array{Float64,1}, μ::Float64, σ²::Float64=1., scale::Float64=0.2)
    return scale * exp.(-1/(σ²+1e-16) .*(x .- μ).^2)
end
@everywhere bathymetry(x::Array{Float64,1}, μ₁::Float64, σ²₁::Float64,μ₂::Float64, σ²₂::Float64) = bathymetry(x, μ₁, σ²₁) .+ bathymetry(x, μ₂, σ²₂)
@everywhere bathymetry(x::Array{Float64,1}, params::Array{Float64,1}) = length(params)>4 ? params : bathymetry(x, params...)


@everywhere function simulation(param, sim_params::simulation_setup, observation::observation_data)
    println("Running on worker $(myid())")
    solver = swe.SWESolver(sim_params.xbounds, sim_params.timestep,
                            sim_params.nx, sim_params.tend, g=sim_params.g,
                            kappa=sim_params.kappa, dealias=sim_params.dealias,
                            tstart=observation.tstart,
                            problemtype=sim_params.scenario);
    sample_bathy = bathymetry(solver.domain.x, param)
    sim_observations, _, _, _ = solver.solve(sample_bathy, sensor_pos=observation.x)
    return sim_observations
end

function add_noise!(observation::Array{Float64,2}, noise_var::Float64)
    noise = zero(observation)
    # Maximum absolute displacement of sensor data
    noise_level = vec(maximum(abs.(observation .- observation[1]), dims=1))

    # percentage of absolute maximum displacement
    noise_var = noise_var * maximum(noise_level)
    noise_dist = MvNormal(zero(noise_level), noise_var.*noise_level)
    noise = rand(noise_dist, size(observation)[1])
    observation[:] = observation + noise'
    return observation
end


function load_observation_data(file_path::String, noise_var::Float64=0.0)
    h5open(file_path, "r") do file
        dt = attrs(file)["dt"]
        tinterval = attrs(file)["T_N"]
        t = collect(0:dt:tinterval)
        b = read(file["b_exact"])
        observation_H = read(file["H_sensor"])

        #obs_itp = [LinearInterpolation(t, H_sensor) for H_sensor in eachcol(observation_H)]
        sensor_pos = [3.5, 5.5, 7.5]
        t_measured = collect(0:0.001:10)
        tid = findall(x->x in t_measured, t)
        observation_H = observation_H[tid, :]
        tstart = attrs(file)["tstart"]
        #observation_H = vcat([obs_itp_i.(t_measured) for obs_itp_i in obs_itp])
        # Add noise to the observation percentage of abs max value
        add_noise!(observation_H, noise_var)
        return observation_data(t_measured, sensor_pos, observation_H, tstart),b
    end
end

config = TOML.parsefile("config.toml")

# Define the simulation parameters
xbounds = config["simulation"]["xbounds"]
timestep = config["simulation"]["timestep"]
nx = config["simulation"]["nx"]
tend = config["simulation"]["tinterval"]
g = config["simulation"]["g"]
kappa = config["simulation"]["kappa"]
dealias = config["simulation"]["dealias"]
scenario = config["simulation"]["scenario"]
problem_bathy = config["simulation"]["bathymetry"]
sim_params = simulation_setup(xbounds, timestep, nx, tend, g, kappa, dealias, scenario)

# Load the observation data
observation_path = config["observation"]["path"]
exp_name = "$(scenario)_$(problem_bathy)"
if problem_bathy == "gaussian"
    exp_name *= "_$(bathy_config["npeaks"])_peaks"
end
sim_path = joinpath(observation_path, exp_name)
observation_file = joinpath(sim_path,"jl_simulation_data.h5")
println("Loading observation data from $(observation_file)")
observation, exact_b = load_observation_data(observation_file, config["observation"]["noise_var"])


# Define the parameter space
inputs = [
    ([3.0,1.0], sim_params, observation),
    ([8.0,0.2], sim_params, observation),
    ([5.0, 1.5], sim_params, observation),
    ([10.0, 1.0], sim_params, observation)
]

# Run the simulation

@time results = pmap(x->simulation(x[1], x[2], x[3]), inputs)

@time for p in inputs
    println("Running on worker $(myid())")
    simulation(p[1], p[2], p[3])
end
println("Done")