using Pkg
Pkg.activate(".")

using Turing
using SliceSampling
using PyCall
using Interpolations
using StatsPlots
using Serialization
# Import the SWESolver class from the utils module
@pyimport swe_wrapper as swe
using HDF5

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

function logjoint(x, model)
    sim_observations = model.forward_model(x)
    log_prior = sum(model.logprior(x))
    log_likelihood = sum(model.ll(sim_observations - model.observation.h))
    return log_prior + log_likelihood
end

function simulation(b, sim_params::simulation_setup, observation::observation_data)
    solver = swe.SWESolver(sim_params.xbounds, sim_params.timestep,
                            sim_params.nx, sim_params.tend, sim_params.g,
                            sim_params.kappa, sim_params.dealias);
    simulation_h, _, t = solver.solve(b);
    x = solver.domain.x
    sim_interp = LinearInterpolation((t,x), simulation_h)
    sim_observations = sim_interp.(observation.t, observation.x')
    return sim_observations
end

function mhsampler(model, n, initial_x; γ=0.1, burn_in=0)
    chain = zeros(n-burn_in, length(initial_x))
    x = initial_x
    logp = logjoint(x, model)
    for i in 1:n
        println("Iteration: ", i)
        x_new = x + rand(Normal(0,γ),(length(x),1))
        display(plot(x_new))
        if maximum(x_new) > 0.28
            logp_new = -Inf
        else
            logp_new = logjoint(x_new, model)
        end
        if log(rand()) < logp_new - logp
            println("Accepted")
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
        obs_interpolated = LinearInterpolation((t, x), h')
        sensor_pos = [2., 4., 6., 8.]
        t_measured = collect(0:0.1:10)
        observation_h = obs_interpolated.(t_measured, sensor_pos')
        #noise_dist = Normal(0, 0.1)
        #noise = rand(noise_dist, size(observation_h))
        #observation_h += noise

        return observation_data(collect(t_measured), sensor_pos, observation_h),b
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
init_b = exact_b[2:2:end-1]
init_b = rand(MvNormal(init_b, 0.01), 1)
display(plot(init_b))

forward_model(b::Matrix) = simulation(b, sim_params, observation)
logprior(b) = sum(logpdf(Product(fill(Uniform(0,0.2), sim_params.nx)), b))
loglikelihood(x) = sum(logpdf(Normal(0, 0.01), x))

my_model = model(logprior, loglikelihood, observation, forward_model)

chain = mhsampler(my_model, 4, init_b; burn_in=0, γ=0.05)
display(plot(chain))
if save
    serialize("./data/results/chain_test_init_noise_mymh.jls", chain)
    #plot(chain)
    #savefig("./plots/mcmc_bathymetry_reco_chain_soph_laplace_long_randpermgibbs.pdf")
    #println(chain)
end