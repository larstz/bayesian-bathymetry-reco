
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
    noise_dist = MvNormal(zero(noise_level), Diagonal((noise_var.*noise_level)).^2)
    noise = rand(noise_dist, size(observation)[1])
    observation[:] = observation + noise'
    return observation
end

export load_observation
function load_observation(file_path::String, noise_var::Float64=0.0)
    file = joinpath(file_path, "jl_simulation_data.h5")
    h5open(file, "r") do file
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

export simulation_setup
struct simulation_setup
    xbounds::Array{Float64, 1}
    timestep::Float64
    nx::Int
    tend::Float64
    g::Float64
    kappa::Float64
    dealias::Float64
    scenario::String
    bathy_name::String
end

export mcmc_setup
struct mcmc_setup
    n::Int
    γ::Union{Float64, Array{Float64, 1}}
    burn_in::Int
    likelihood_σ::Float64
    initial_θ::Union{Array{Float64, 1}, Array{Array{Float64, 1}, 1}}
end

export observation_settings
struct observation_settings
    path::String
    noise_var::Float64
end

export io_settings
struct io_setup
    save::Bool
    output_dir::String
end

export reconstructor
struct reconstructor
    sim_params::simulation_setup
    mcmc_params::mcmc_setup
    obs_settings::observation_settings
    io_settings::io_setup
end

export bathymetry_setup
struct bathymetry_setup
    θ::Array{Float64,1}
end

export load_config
function load_config(file_path::String)
    config = TOML.parsefile(file_path)
    sim_params = read_simulation_parameters(config["simulation"])
    mcmc_params = read_mcmc_parameters(config["sampler"])
    obs_settings = read_observation_settings(config["observation"])
    io_settings = read_io_settings(config["output"])
    return reconstructor(sim_params, mcmc_params, obs_settings, io_settings)
end

export read_simulation_parameters
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

export read_mcmc_parameters
function read_mcmc_parameters(config::Dict{String,Any})
    n = config["n_samples"]
    γ = config["stepsize"]
    burn_in = config["burn_in"]
    likelihood_σ = config["likelihood_var"]
    init = config["initial"]
    mcmc_params = mcmc_setup(n, γ, burn_in, likelihood_σ,init)
    return mcmc_params
end

export read_observation_settings
function read_observation_settings(config::Dict{String,Any})
    path = config["path"]
    noise_var = config["noise_var"]
    obs_settings = observation_settings(path, noise_var)
    return obs_settings
end

export read_io_settings
function read_io_settings(config::Dict{String,Any})
    save = config["save"]
    path = config["path"]
    io_settings = io_setup(save, path)
    return io_settings
end

export read_bathymetry_parameters
function read_bathymetry_parameters(config::Dict{String,Any})
    μ = config["μ"]
    σ² = config["σ²"]
    scale = config["scale"]
    bathy_params = bathymetry_params(μ, σ², scale)
    return bathy_params
end
