using Pkg
Pkg.activate(".")

using Turing
using PyCall
using Interpolations
using HDF5

# Import the SWESolver class from the utils module
@pyimport utils as pyutils

struct observation_data
    t::Array{Float64}
    x::Array{Float64}
    h::Array{Float64}
end

# Define the Turing model
@model function shallow_water_model(observation::observation_data)
    # Define the prior for the bathymetry peak
    peak ~ Uniform(0, 10)

    # Create an instance of the SWESolver
    xbounds = (0., 10.)
    nx = 64
    tend = 10
    timestep = 1e-3
    g = 9.81
    kappa = 0.2
    dealias = 3/2

    # Create SWE solver and calculate the solution
    solver = pyutils.SWESolver(xbounds, timestep, nx, tend, g, kappa, dealias)
    # Use the solver to simulate the shallow water equation with the sampled peak
    simulation_h, _, t = solver.solve(peak)
    x = solver.domain.x


    sim_interp = LinearInterpolation((t,x), simulation_h)

    # interpolate values of the simulation to match the observation
    sim_observations = sim_interp.(observation.t, observation.x')

    # Compare the simulation with the observation
    for i in eachindex(observation.h)
        observation.h[i] ~ Normal(sim_observations[i], 0.1)
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
        sensor_pos = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
        t_measured = collect(0:0.1:10)
        observation_h = obs_interpolated.(t_measured, sensor_pos')

        return observation_data(collect(t_measured), sensor_pos, observation_h)
    end
end

file_path = "./data/toy_measurement/simulation_data.h5"
observation = load_observation_data(file_path)

# Instantiate the model
model = shallow_water_model(observation)

# Sample from the posterior
chain = sample(model, MH(), 3)
# Because the solver is written in python we need a gradient free sampler like MH
# Print the results
println(chain)