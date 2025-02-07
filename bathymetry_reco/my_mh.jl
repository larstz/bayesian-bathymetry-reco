using Pkg
Pkg.activate(".")

using Turing
using TOML
using Dates
using PyCall
using Interpolations
using StatsPlots
using Serialization
# Import the SWESolver class from the utils module
@pyimport swe_wrapper as swe
using HDF5
using ProgressBars

struct observation_data
    t::Array{Float64}
    x::Array{Float64}
    h::Array{Float64}
end

struct simulation_setup
    xbounds::Array{Float64, 1}
    timestep::Float64
    nx::Int
    tend::Float64
    g::Float64
    kappa::Float64
    dealias::Float64
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

bathymetry(x::Array{Float64,1}, params::Array{Float64,1}) = length(params)>3 ? params : bathymetry(x, params...)

function logjoint(x, observation::observation_data)
    log_prior = logprior(x)
    if log_prior == -Inf
        return -Inf
    end
    sim_observations = forward_model(x)
    log_likelihood = sum(loglikelihood(sim_observations - observation.h))
    return log_prior + log_likelihood
end

function simulation(param, sim_params::simulation_setup, observation::observation_data)
    solver = swe.SWESolver(sim_params.xbounds, sim_params.timestep,
                            sim_params.nx, sim_params.tend, sim_params.g,
                            sim_params.kappa, sim_params.dealias);
    sample_bathy = bathymetry(solver.domain.x, param)
    simulation_h, _, t = solver.solve(sample_bathy)
    simulation_H = simulation_h .+ sample_bathy'
    x = solver.domain.x
    sim_interp = LinearInterpolation((t,x), simulation_H)
    sim_observations = sim_interp.(observation.t, observation.x')
    return sim_observations
end

function mhsampler(observation, n, initial_x; γ=0.1, burn_in=0)
    chain = zeros(n-burn_in, length(initial_x))
    x = initial_x
    logp = logjoint(x, observation)
    for i in ProgressBar(1:n)
        x_new = x + rand(Normal(0,γ), size(x))
        # if x_new[2] < 0 || x_new[1] < 0 || x_new[1] > 10
        #     logp_new = -Inf
        # else
            logp_new = logjoint(x_new, observation)
        # end
        if rand() < exp(logp_new - logp)
            x = x_new
            logp = logp_new
        end
        if i > burn_in
            chain[i-burn_in, :] = x
        end
    end
    return chain
end


function load_observation_data(file_path::String, noise_var::Float64=0.0)
    h5open(file_path, "r") do file
        t = read(file["t_array"])
        x = read(file["xgrid"])
        h = read(file["h"])
        b = read(file["b_exact"])
        H = h .+ b
        obs_interpolated = LinearInterpolation((t, x), H')
        sensor_pos = [2., 4., 6., 8.]
        t_measured = collect(0:0.1:10)
        observation_H = obs_interpolated.(t_measured, sensor_pos')
        noise = noise_var > 0 ? rand(Normal(0, noise_var), size(observation_H)) : zeros(size(observation_H))
        observation_H += noise
        return observation_data(collect(t_measured), sensor_pos, observation_H),b
    end
end

config = TOML.parsefile("config.toml")

# Define the simulation parameters
xbounds = config["simulation"]["xbounds"]
timestep = config["simulation"]["timestep"]
nx = config["simulation"]["nx"]
tend = config["simulation"]["tend"]
g = config["simulation"]["g"]
kappa = config["simulation"]["kappa"]
dealias = config["simulation"]["dealias"]
sim_params = simulation_setup(xbounds, timestep, nx, tend, g, kappa, dealias)

save = config["output"]["save"]
target_dir = config["output"]["path"]*Dates.format(now(), "Y-mm-dd-HH-MM-SS")

observation, exact_b = load_observation_data(config["observation"]["path"], config["observation"]["noise_var"])

# Instantiate the model
#init_b = exact_b[2:2:end-1]
#init_b = rand(MvNormal(init_b, 0.01), 1)
#display(plot(init_b))

# Define parameters for MH sampler

γ = config["sampler"]["stepsize"] # stepsize
likelihood_σ = config["sampler"]["likelihood_var"] # likelihood variance
n_samples = config["sampler"]["n_samples"] # number of samples
burnin = config["sampler"]["burnin"] # burnin

forward_model(params) = simulation(params, sim_params, observation)

logprior(params::Array{Float64,1}) = length(params)>3 ? sum(logpdf(Uniform(0,0.21),params)) : logprior(params...)
logprior(μ::Float64) = logpdf(Uniform(0, 10), μ)
logprior(μ::Float64, σ²::Float64) = logprior(μ) + logpdf(Uniform(0, 2),σ²)
logprior(μ::Float64, σ²::Float64, s::Float64) = logprior(μ, σ²) + logpdf(Uniform(0.1, 0.25), s)
loglikelihood(x::Array{Float64}) = logpdf(Normal(0, likelihood_σ), x)

if config["sampler"]["parametrized"]
    init_b = config["sampler"]["initial"]
else
    init_b = exact_b[2:2:end-1]
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
    plot(chain; label=["μ" "σ²" "scale"], title="Chain for μ₀=4.5 and σ²₀=0.5 scale=0.2", xlabel="Iteration", ylabel="Value")
    mkpath("./plots")
    savefig("./plots/chain.pdf")
end