using Pkg
Pkg.activate(".")
using Distributions
using TOML
using Dates
using PyCall
using Interpolations
using Plots
using Serialization
# Import the SWESolver class from the utils module
@pyimport swe_wrapper as swe
using HDF5
using ProgressBars

struct observation_data
    t::Array{Float64}
    x::Array{Float64}
    H::Array{Float64}
    tstart::Float64
end

struct simulation_setup
    xbounds::Array{Float64, 1}
    timestep::Float64
    nx::Int
    tend::Float64
    g::Float64
    kappa::Float64
    dealias::Float64
    scenario::String
end

struct model
    logprior
    ll
    observation
    forward_model
end

function bathymetry(x::Array{Float64,1}, μ::Float64, σ²::Float64=1., scale::Float64=0.2)
    return scale * exp.(-1/(σ²+1e-16) .*(x .- μ).^2)
end

bathymetry(x::Array{Float64,1}, μ₁::Float64, σ²₁::Float64,μ₂::Float64, σ²₂::Float64) = bathymetry(x, μ₁, σ²₁) .+ bathymetry(x, μ₂, σ²₂)
bathymetry(x::Array{Float64,1}, params::Array{Float64,1}) = length(params)>4 ? params : bathymetry(x, params...)

function logjoint(x, observation::observation_data)
    log_prior = logprior(x)
    if log_prior == -Inf
        return -Inf
    end
    sim_observations = forward_model(x)
    try
        log_likelihood = sum(loglikelihood(sim_observations - observation.H))
        return log_prior + log_likelihood
    catch err
        if isa(err, DimensionMismatch)
            return -Inf
        else
            rethrow(err)
        end
    end
end

function simulation(param, sim_params::simulation_setup, observation::observation_data)
    solver = swe.SWESolver(sim_params.xbounds, sim_params.timestep,
                            sim_params.nx, sim_params.tend, g=sim_params.g,
                            kappa=sim_params.kappa, dealias=sim_params.dealias,
                            tstart=observation.tstart,
                            problemtype=sim_params.scenario);
    sample_bathy = bathymetry(solver.domain.x, param)
    sim_observations, _, _, _ = solver.solve(sample_bathy, sensor_pos=observation.x)
    return sim_observations
end

function mhsampler(observation, n, initial_x; γ=0.1, burn_in=0)
    chain = zeros(n-burn_in, length(initial_x)+1)
    x = initial_x
    logp = logjoint(x, observation)
    for i in ProgressBar(1:n)
        x_new = x + γ .* rand(Normal(0,1), size(x))

        logp_new = logjoint(x_new, observation)

        if rand() < exp(logp_new - logp)
            x = x_new
            logp = logp_new
        end
        if i > burn_in
            chain[i-burn_in, :] = [x..., logp]
        end
    end
    return chain
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

#TODO: Pass the config file as an argument in the command line
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

# Set the output directory
save = config["output"]["save"]
target_dir = joinpath(config["output"]["path"],"$(exp_name)_"*Dates.format(now(), "Y-mm-dd-HH-MM-SS"))

# Define parameters for MH sampler

γ = config["sampler"]["stepsize"] # stepsize
likelihood_σ = config["sampler"]["likelihood_var"] # likelihood variance
n_samples = config["sampler"]["n_samples"] # number of samples
burnin = config["sampler"]["burnin"] # burnin

forward_model(params) = simulation(params, sim_params, observation)

logprior(params::Array{Float64,1}) = length(params)>4 ? sum(logpdf(Uniform(0,0.21),params)) : logprior(params...)
logprior(μ::Float64) = logpdf(Uniform(xbounds[1], xbounds[2]), μ)
logprior(μ::Float64, σ²::Float64) = logprior(μ) + logpdf(Uniform(0, 2),σ²)
logprior(μ₁::Float64, σ²₁::Float64,μ₂::Float64, σ²₂::Float64) = logprior(μ₁, σ²₁) + logprior(μ₂, σ²₂)
logprior(μ::Float64, σ²::Float64, s::Float64) = logprior(μ, σ²) + logpdf(Uniform(0.1, 0.25), s)
loglikelihood(x::Array{Float64}) = logpdf(Normal(0, likelihood_σ), x)

if config["sampler"]["parametrized"]
    init_b = config["sampler"]["initial"]
    println("Using parametrized initial value: $init_b")
else
    #TODO implment initialization for discretized sampling of bathymetry
    init_b = exact_b
end

chain = mhsampler(observation, n_samples, init_b; burn_in=burnin, γ=γ)

if save
    mkpath(target_dir)
    cd(target_dir)

    # Serialize the chain
    serialize("chain_p.jls", chain)

    # store the configuration file for reproducibility
    open("config_copy.toml", "w") do io
        TOML.print(io, config)
    end
    # Plot the chain
    title_init = config["sampler"]["parametrized"] ? "p₀=$(init_b)" : "p₀=exact"
    plot(chain[:,1:end-1]; label=["μ" "σ²" "scale"], title="Chain for $(title_init)", xlabel="Iteration", ylabel="Value")
    mkpath("./plots")
    savefig("./plots/chain.pdf")
    plot(chain[:,end]; label="log p(θ)", title="Chain logp for $(title_init)", xlabel="Iteration", ylabel="Value")
    savefig("./plots/logp.pdf")
end