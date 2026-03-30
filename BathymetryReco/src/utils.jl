
export ObservationData

"""
    ObservationData(t, x, sim_x, H, tstart, noise_std)

Data structure to hold the observation data for the bathymetry reconstruction problem.

# Fields
- `t`: Time points of the observations.
- `x`: Sensor positions corresponding to the observations.
- `sim_x`: Spatial grid points used in the simulation.
- `H`: Observed water surface heights at the sensor positions and time points.
- `tstart`: Starting time of the observations.
- `noise_std`: Standard deviation of the noise in the observations.
"""
struct ObservationData
    t::Array{Float64}
    x::Array{Float64}
    sim_x::Array{Float64}
    H::Array{Float64}
    tstart::Float64
    noise_std::Array{Float64}
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

export get_perc_noise
function get_perc_noise(observation::Array{Float64,2}, noise_var::Float64)
    noise = zero(observation)
    # Maximum absolute displacement of sensor data
    noise_level = vec(maximum(abs.(observation .- observation[1]), dims=1))

    # percentage of absolute maximum displacement
    noise_dist = MvNormal(zero(noise_level), Diagonal((noise_var.*noise_level)).^2)
    noise = rand(noise_dist, size(observation)[1])'
    return noise
end

export load_toy_observation, load_observation

"""
    load_toy_observation(file_path::String, noise_var::Float64=0.0; sensor_id::Array{Int64}=[2, 3, 4], sensor_rate::Float64=0.001)

Load simulated observation data into an `ObservationData` struct from a file, with options
for adding noise and selecting specific sensors and sensor rates.

# Arguments
- `file_path::String`: Path to the file containing the simulated observation data
- `noise_var::Float64`: Variance of the noise to be added to the observations

# Keywords
- `sensor_id::Array{Int64}=[2, 3, 4]`: Array of sensor IDs to select from the data
- `sensor_rate::Float64`: Rate at which sensors measure the observations (default is 0.001)
"""
function load_toy_observation(file_path::String, noise_var::Float64=0.0; sensor_id::Array{Int64}=[2, 3, 4], sensor_rate::Float64=0.001)
    id2pos = [3.5, 5.5, 7.5]
    sensor_id = sensor_id .- 1 # convert to 1-based indexing (only Sensors 2, 3, 4 are stored in the file)
    file = joinpath(file_path, "jl_simulation_data.h5")
    h5open(file, "r") do file
        dt = attrs(file)["dt"]
        tinterval = attrs(file)["T_N"]
        t = collect(0:dt:tinterval)
        b = read(file["b_exact"])
        observation_H = read(file["H_sensor"])
        x = read(file["xgrid"])

        t_measured = collect(0:sensor_rate:10)
        tid = findall(x->x in t_measured, t)
        observation_H = observation_H[tid, sensor_id]
        tstart = attrs(file)["tstart"]
        noise = get_perc_noise(observation_H, noise_var)
        observation_H = observation_H + noise
        noise_std = std(noise, dims=1)
        sensor_pos = id2pos[sensor_id]
        return ObservationData(t_measured, sensor_pos, x, observation_H, tstart, noise_std),b
    end
end

"""
    load_observation(file_path::String, t_start::Float64, t_interval::Float64; sensor_id::Array{Int64}=[2, 3, 4], noise_var::Float64=0.0)

Load real observation data into an `ObservationData` struct from a file, with options for
adding noise and selecting specific sensors.

# Arguments
- `file_path::String`: Path to the file containing the real observation data
- `t_start::Float64`: Starting time of the observations to be loaded
- `t_interval::Float64`: Time interval for which to load the observations

# Keywords
- `sensor_id::Array{Int64}=[2, 3, 4]`: Array of sensor IDs to select from the data
- `noise_var::Float64`: Variance of the noise to be added to the observations
"""
function load_observation(file_path::String, t_start::Float64, t_interval::Float64; sensor_id::Array{Int64}=[2, 3, 4], noise_var::Float64=0.0)
    measurement = CSV.read(file_path, DataFrame)
    id2pos = [3.5, 5.5, 7.5]
    sensor_id = sensor_id .- 1 # convert to 1-based indexing
    sensor_pos = id2pos[sensor_id]
    # get noise information
    baseline_id = measurement.Time.<t_start
    base_measurement = Matrix(measurement[baseline_id,r"Sensor[2-4]"])./100 #convert cm to m
    noise_std = std(base_measurement, dims=1)

    # extract relevant measurement data
    obs_id = t_start.<=measurement.Time.<=t_start+t_interval
    observation = measurement[obs_id, :]
    t = round.(Vector(observation[:,"Time"]).-t_start, digits=2)
    observation_H = Matrix(observation[:,r"Sensor[2-4]"])./100 .+0.3 # add 30 cm to all measurements, convert cm to m
    observation_H = observation_H[:, sensor_id] # select only the relevant sensors

    # add noise
    if noise_var > 0.0
        noise = get_perc_noise(observation_H, noise_var)
        observation_H = observation_H + noise
    end

    return ObservationData(t, sensor_pos, [0.], observation_H, t_start, noise_std)
end

export SimulationSetup
"""
    SimulationSetup(xbounds, sensor_pos, timestep, nx, tstart, tinterval, g, kappa, dealias, scenario, bc_file)

Data structure to hold the simulation parameters for the shallow water equations solver.

# Fields
- `xbounds`: Spatial bounds of the simulation domain
- `sensor_pos`: Positions of the sensors in the simulation domain
- `timestep`: Time step for the simulation
- `nx`: Number of spatial grid points in the simulation
- `tstart`: Starting time of the simulation
- `tinterval`: Total time interval for the simulation
- `g`: Gravitational acceleration constant
- `kappa`: friction parameter for the shallow water equations
- `dealias`: Dealiasing factor for the spectral solver
- `scenario`: String identifier for the simulation scenario
- `bc_file`: Path to the file containing boundary condition data for the simulation
- `bathy_name`: String identifier for the type of bathymetry used in the simulation
"""
struct SimulationSetup
    xbounds::Array{Float64, 1}
    sensor_pos::Array{Float64, 1}
    timestep::Float64
    nx::Int
    tstart::Float64
    tinterval::Float64
    g::Float64
    kappa::Float64
    dealias::Float64
    scenario::String
    bc_file::String
    bathy_name::String
end

export PriorSettings
"""
    PriorSettings(type, lengthscale, var, loc, scale)

Data structure to hold the settings for the prior distribution used in the MCMC sampling.

# Fields
- `type`: Array of strings specifying the type(s) of prior distribution(s) (e
"""
struct PriorSettings
    type:: Array{String,1}
    lengthscale::Float64
    var::Float64
    loc::Union{Float64, Array{Float64, 1}}
    scale::Union{Float64, Array{Float64, 1}}
end

export ProposalSettings
struct ProposalSettings
    type::String
    kernel::String
    lengthscale::Float64
    var::Float64
end

export MCMCSetup
struct MCMCSetup
    n::Int
    dim::Int
    n_chains::Int
    γ::Union{Float64, Array{Float64, 1}}
    burn_in::Int
    likelihood_σ::Union{Float64, Array{Float64, 1}}
    prior::Union{PriorSettings, Nothing}
    proposal::Union{ProposalSettings, Nothing}
    initial_θ::Union{Array{Float64, 1}, Array{Array{Float64, 1}, 1}, Array{Union{}, 1}}
end

export ObservationSettings
struct ObservationSettings
    path::String
    real_data::Bool
    noise_var::Float64
    sensor_rate::Float64
    sensor_id::Array{Int64, 1}
end

export IOSettings
struct IOSettings
    save::Bool
    output_dir::String
end

export Reconstructor
struct Reconstructor
    sim_params::SimulationSetup
    mcmc_params::MCMCSetup
    obs_settings::ObservationSettings
    io_settings::IOSettings
end

export BathymetrySetup
struct BathymetrySetup
    θ::Array{Float64,1}
    n_peaks::Int
end

export load_config
function load_config(file_path::String)
    config = TOML.parsefile(file_path)
    sim_params = read_simulation_parameters(config["simulation"])
    mcmc_params = read_mcmc_parameters(config["sampler"])
    obs_settings = read_observation_settings(config["observation"])
    io_settings = read_io_settings(config["output"])
    return Reconstructor(sim_params, mcmc_params, obs_settings, io_settings)
end

function load_config(config::Dict{String,Any})
    sim_params = read_simulation_parameters(config["simulation"])
    mcmc_params = read_mcmc_parameters(config["sampler"])
    obs_settings = read_observation_settings(config["observation"])
    io_settings = read_io_settings(config["output"])
    return Reconstructor(sim_params, mcmc_params, obs_settings, io_settings)
end

export read_simulation_parameters
function read_simulation_parameters(config::Dict{String,Any})
    xbounds = config["xbounds"]
    sensor_pos = config["sensor_position"]
    timestep = config["timestep"]
    nx = get(config, "nx", 64)
    tstart = config["tstart"]
    tend = config["tinterval"]
    g = config["g"]
    kappa = config["kappa"]
    dealias = config["dealias"]
    scenario = config["scenario"]
    bc_file = config["bc_file"]
    problem_bathy = config["bathymetry"]
    sim_params = SimulationSetup(xbounds, sensor_pos, timestep, nx, tstart, tend, g, kappa, dealias, scenario, bc_file, problem_bathy)
    return sim_params
end

export read_mcmc_parameters
function read_mcmc_parameters(config::Dict{String,Any})
    n = config["n_samples"]
    dim = get(config, "nx", 64)
    n_chains = config["n_chains"]
    γ = config["stepsize"]
    burn_in = config["burn_in"]
    likelihood_σ = config["likelihood_var"]
    init = config["initial"]
    # Convert Vector{Any} to Vector{Vector{Float64}} for Julia 1.12 TOML compatibility
    if init isa Vector{Any}
        init = Vector{Float64}[Vector{Float64}(x) for x in init]
    end
    prior_settings = read_prior_settings(get(config, "prior", nothing))
    proposal_settings = read_proposal_settings(get(config, "proposal", nothing))
    mcmc_params = MCMCSetup(n, dim, n_chains, γ, burn_in, likelihood_σ, prior_settings, proposal_settings, init)
    return mcmc_params
end

export read_prior_settings
function read_prior_settings(config::Union{Dict{String,Any}, Nothing})
    if config === nothing
        return nothing
    end
    type = config["type"] isa String ? [config["type"]] : config["type"]
    lengthscale = get(config, "lengthscale", 0.0)
    var = get(config, "var", 0.0)
    loc = get(config, "loc", 0.0)
    scale = get(config, "scale", 1.0)
    return PriorSettings(type, lengthscale, var, loc, scale)
end

read_prior_settings(config::String) = nothing
read_proposal_settings(config::String) = nothing

export read_proposal_settings
function read_proposal_settings(config::Union{Dict{String,Any}, Nothing})
    if config === nothing
        return nothing
    end
    type = config["type"]
    kernel = get(config, "kernel", "")
    lengthscale = get(config, "lengthscale", 0.0)
    var = get(config, "var", 0.0)
    return ProposalSettings(type, kernel, lengthscale, var)
end

export read_observation_settings
function read_observation_settings(config::Dict{String,Any})
    path = config["path"]
    real_data = config["real_data"]
    noise_var = config["noise_var"]
    sensor_id = config["sensor_id"]
    sensor_rate = config["sensor_rate"]
    obs_settings = ObservationSettings(path, real_data, noise_var, sensor_rate, sensor_id)
    return obs_settings
end

export read_io_settings
function read_io_settings(config::Dict{String,Any})
    save = config["save"]
    path = config["path"]
    io_settings = IOSettings(save, path)
    return io_settings
end

export read_bathymetry_parameters
function read_bathymetry_parameters(config::Dict{String,Any})
    params = config["parameters"]
    npeaks = config["npeaks"]
    bathy_params = BathymetrySetup(params, npeaks)
    return bathy_params
end
