using Pkg
Pkg.activate(".")

using Turing
using PyCall
using Interpolations
using StatsPlots
using Serialization
# Import the SWESolver class from the utils module
@pyimport swe_wrapper as swe
using HDF5 # Has to imported after my module otherwise pycall cannot find swe_wrapper

struct observation_data
    t::Array{Float64}
    x::Array{Float64}
    h::Array{Float64}
end

# Define the Turing model
@model function shallow_water_model(observation::observation_data)
    # Define the prior for the bathymetry peak
    println("Running the model")
    #peak ~ Uniform(0, 10)
    #s ~ Uniform(0.001, 2)
    # Create SWE solver
    b ~ MvNormal(zeros(64), 1)
    xbounds = (0., 10.)
    nx = 64
    tend = 10
    timestep = 1e-3
    g = 9.81
    kappa = 0.2
    dealias = 3/2

    # Create SWE solver and calculate the solution
    solver = swe.SWESolver(xbounds, timestep, nx, tend, g, kappa, dealias);
    # Use the solver to simulate the shallow water equation with the sampled peak
    #simulation_h, _, t = solver.solve((peak, s));
    simulation_h, _, t = solver.solve(b);
    x = solver.domain.x
    sim_interp = LinearInterpolation((t,x), simulation_h)

    # interpolate values of the simulation to match the observation
    sim_observations = sim_interp.(observation.t, observation.x')

    # Compare the simulation with the observation
    for i in 1:length(observation.t)
        observation.h[i,:] ~ MvNormal(sim_observations[i,:], 0.01)
    end
end

# Example usage
# Assuming `observation` is your observed data

# Load the observation data from the HDF5 file
function load_observation_data(file_path::String)
    h5open(file_path, "r") do file
        t = read(file["t_array"])
        x = read(file["xgrid"])
        h = read(file["h"])

        obs_interpolated = LinearInterpolation((t, x), h')
        sensor_pos = [2., 4., 6., 8.]
        t_measured = collect(0:0.1:10)
        observation_h = obs_interpolated.(t_measured, sensor_pos')
        noise_dist = Normal(0, 0.1)
        noise = rand(noise_dist, size(observation_h))
        observation_h += noise

        return observation_data(collect(t_measured), sensor_pos, observation_h)
    end
end

file_path = "./data/toy_measurement/simulation_data.h5"
observation = load_observation_data(file_path)

# Instantiate the model
model = shallow_water_model(observation)

# Sample from the posterior
chain = sample(model, MH(), 50, burnin=10)
# Because the solver is written in python we need a gradient free sampler like MH
# Print the results

serialize("./data/results/chain_soph.jls", chain)
plot(chain)
savefig("./plots/mcmc_bathymetry_reco_chain_soph.pdf")
println(chain)