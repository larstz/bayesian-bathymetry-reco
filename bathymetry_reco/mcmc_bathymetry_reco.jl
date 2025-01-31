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
using HDF5 # Has to imported after my module otherwise pycall cannot find swe_wrapper

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

# Define the Turing model
@model function shallow_water_model(observation::observation_data, sim_params::simulation_setup;
                                    sigma::Matrix{Float64}=Matrix(I, 64, 64))
    # Define the prior for the bathymetry peak
    println("Running the model")
    #peak ~ Uniform(0, 10)
    #s ~ Uniform(0.001, 2)

    #b ~ MvNormal(zeros(64), 0.1*sigma)
    b ~ Product(fill(Uniform(0,0.2), sim_params.nx))
    #display(plot(b, label=""))
    if maximum(b) > 0.28
        Turing.@addlogprob! -Inf
        return nothing
    end
    # Create SWE solver and calculate the solution
    solver = swe.SWESolver(sim_params.xbounds, sim_params.timestep,
                            sim_params.nx, sim_params.tend, sim_params.g,
                            sim_params.kappa, sim_params.dealias);
    # Use the solver to simulate the shallow water equation with the sampled peak
    #simulation_h, _, t = solver.solve((peak, s));
    simulation_h, _, t = solver.solve(b);
    x = solver.domain.x
    sim_interp = LinearInterpolation((t,x), simulation_h)
    if maximum(t)<10
        println("Not simulated until end")
        display(heatmap(simulation_h, aspect_ratio=1))
        readline()
    end
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

function create_laplace_matrix_1d(N)
    # N ist die Anzahl der Gitterpunkte
    L = zeros(Float64, N, N)  # Erstellen einer N×N Matrix

    # Füllen der Matrix mit den Werten für den Laplace-Operator
    for i in 1:N
        if i > 1
            L[i, i - 1] = 1  # Nachbar links
        end
        L[i, i] = -2  # Zentraler Punkt
        if i < N
            L[i, i + 1] = 1  # Nachbar rechts
        end
    end
    return L
end

save = false

file_path = "./data/toy_measurement/simulation_data.h5"
observation, exact_b = load_observation_data(file_path)

# Define the simulation parameters
xbounds = (0., 10.)
nx =  64
tend = 10
timestep = 1e-3
g = 9.81
kappa = 0.2
dealias = 3/2
sim_params = simulation_setup(xbounds, timestep, nx, tend, g, kappa, dealias)

# Create the covariance matrix
L = create_laplace_matrix_1d(nx)
sigma = 1/nx^2 .*inv(L)inv(L)'

# Instantiate the model
model = shallow_water_model(observation, sim_params; sigma=sigma)
init_b = exact_b[2:2:end-1]
init_b = rand(MvNormal(init_b, 0.01), 1)
display(plot(init_b))
readline()
# Sample from the posterior
n_samples = 10
num_warmup = 0
chain = sample(model, MH(0.1*I(nx)), n_samples, num_warmup=num_warmup, initial_params=init_b)
# Because the solver is written in python we need a gradient free sampler like MH
# Print the results
if save
    serialize("./data/results/chain_test_init_noise_turing.jls", chain)
    #plot(chain)
    #savefig("./plots/mcmc_bathymetry_reco_chain_soph_laplace_long_randpermgibbs.pdf")
    #println(chain)
end
