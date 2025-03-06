
export observation_data
struct observation_data
    t::Array{Float64}
    x::Array{Float64}
    H::Array{Float64}
    tstart::Float64
end

export add_noise!
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

export load_observation_data
function load_observation_data(file_path::String, noise_var::Float64=0.0)
    h5open(file_path, "r") do file
        dt = attrs(file)["dt"]
        tinterval = attrs(file)["T_N"]
        t = collect(0:dt:tinterval)
        b = read(file["b_exact"])
        observation_H = read(file["H_sensor"])

        sensor_pos = [3.5, 5.5, 7.5]
        t_measured = collect(0:0.001:10)
        tid = findall(x->x in t_measured, t)
        observation_H = observation_H[tid, :]
        tstart = attrs(file)["tstart"]
        add_noise!(observation_H, noise_var)
        return observation_data(t_measured, sensor_pos, observation_H, tstart),b
    end
end

export simulation_setup,
       mcmc_setup,
       reconstructor,
       read_config_file,
       read_simulation_parameters,
       read_mcmc_parameters

struct simulation_setup
    xbounds::Array{Float64, 1}
    timestep::Float64
    nx::Int
    tend::Float64
    g::Float64
    kappa::Float64
    dealias::Float64
    scenario::String
    bathy_case::String
end

struct mcmc_setup
    n::Int
    γ::Float64
    burn_in::Int
    init::Array{Float64, 1}
end

struct reconstructor
    sim_params::simulation_setup
    mcmc_params::mcmc_setup
end

struct bathymetry_setup
    θ::Array{Float64,1}
end

function read_config_file(file_path::String)
    config = TOML.parsefile(file_path)
    sim_params = read_simulation_parameters(config["simulation"])
    mcmc_params = read_mcmc_parameters(config["mcmc"])
    io_settings = read
    return reconstructor(sim_params, mcmc_params)
end

function read_simulation_parameters(config::Dict{String,Any})
    xbounds = config["xbounds"]
    timestep = config["timestep"]
    nx = config["nx"]
    tend = config["tinterval"]
    g = config["g"]
    kappa = config["kappa"]
    dealias = config["dealias"]
    scenario = config["scenario"]
    problem_bathy = config["bathymetry"]
    sim_params = simulation_setup(xbounds, timestep, nx, tend, g, kappa, dealias, scenario, problem_bathy)
    return sim_params
end

function read_mcmc_parameters(config::Dict{String,Any})
    n = config["n_samples"]
    γ = config["stepsize"]
    burn_in = config["burn_in"]
    init = config["initial"]
    mcmc_params = mcmc_setup(n, γ, burn_in, init)
    return mcmc_params
end

function read_bathymetry_parameters(config::Dict{String,Any})
    μ = config["μ"]
    σ² = config["σ²"]
    scale = config["scale"]
    bathy_params = bathymetry_params(μ, σ², scale)
    return bathy_params
end

