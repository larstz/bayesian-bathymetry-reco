using Pkg
Pkg.activate(".")

using Turing
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
    xbounds::Tuple{Float64, Float64}
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

function bathymetry(x::Array{Float64,1}, params::Tuple{Float64, Float64, Float64})
    return params[3] * exp.(-params[2] * (x .- params[1]).^2)
end

function logjoint(x, model)
    log_prior = sum(model.logprior(x))
    if log_prior == -Inf
        return -Inf
    end
    sim_observations = model.forward_model(x)
    log_likelihood = sum(model.ll(sim_observations - model.observation.h))
    return log_prior + log_likelihood
end

function simulation(param, sim_params::simulation_setup, observation::observation_data)
    solver = swe.SWESolver(sim_params.xbounds, sim_params.timestep,
                            sim_params.nx, sim_params.tend, sim_params.g,
                            sim_params.kappa, sim_params.dealias);
    sample_bathy = bathymetry(solver.domain.x, Tuple(param))
    simulation_h, _, t = solver.solve(sample_bathy)
    simulation_H = simulation_h .+ sample_bathy'
    x = solver.domain.x
    sim_interp = LinearInterpolation((t,x), simulation_H)
    sim_observations = sim_interp.(observation.t, observation.x')
    return sim_observations
end

function mhsampler(model, n, initial_x; γ=0.1, burn_in=0)
    chain = zeros(n-burn_in, length(initial_x))
    x = initial_x
    logp = logjoint(x, model)
    for i in ProgressBar(1:n)
        x_new = x + rand(Normal(0,γ), size(x))
        if x_new[2] < 0 || x_new[1] < 0 || x_new[1] > 10
            logp_new = -Inf
        else
            logp_new = logjoint(x_new, model)
        end
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


function load_observation_data(file_path::String)
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
        noise_dist = Normal(0, 0.01)
        noise = rand(noise_dist, size(observation_H))
        observation_H += noise
        plot(x,H[:,1], label="H")
        plot!(x,b, label="b")
        display(scatter!(sensor_pos, observation_H[1,:], label="Observed"))
        return observation_data(collect(t_measured), sensor_pos, observation_H),b
    end
end

# Define the simulation parameters
xbounds = (0., 10.)
nx =  64
tend = 10
timestep = 1e-3
g = 9.81
kappa = 0.2
dealias = 3/2
sim_params = simulation_setup(xbounds, timestep, nx, tend, g, kappa, dealias)

save = true

file_path = "./data/toy_measurement/simulation_data.h5"
observation, exact_b = load_observation_data(file_path)

# Instantiate the model
#init_b = exact_b[2:2:end-1]
#init_b = rand(MvNormal(init_b, 0.01), 1)
#display(plot(init_b))

forward_model(params) = simulation(params, sim_params, observation)
logprior(x) = sum(logpdf(Uniform(0, 10), x[1])) + sum(logpdf(Uniform(0, 2), x[2]) + logpdf(Uniform(0.1, 0.25), x[3]))
loglikelihood(x) = sum(logpdf(Normal(0, 0.01), x))

my_model = model(logprior, loglikelihood, observation, forward_model)

init_b = [4.5,0.5, 0.1]
n_samples = 1000

chain = mhsampler(my_model, n_samples, init_b; burn_in=0, γ=0.1)
display(plot(chain))
if save
    serialize("./data/results/chain_test_p_s_s_mymh_noisy_obs_H_long.jls", chain)
    plot(chain, label=["μ" "σ²" "scale"], title="Chain for μ₀=4.5 and σ²₀=0.5 scale=0.25", xlabel="Iteration", ylabel="Value")
    savefig("./plots/chain_test_p_s_s_mymh_noisy_obs_H_long.pdf")
    #println(chain)
end