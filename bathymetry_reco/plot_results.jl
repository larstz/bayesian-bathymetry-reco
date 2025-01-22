using Pkg
Pkg.activate(".")
using Turing
using StatsPlots
using Serialization
using PyCall
@pyimport swe_wrapper as swe
using HDF5
struct observation_data
    t::Array{Float64}
    x::Array{Float64}
    b::Array{Float64}
    h::Array{Float64}
end

function load_observation_data(file_path::String)
    h5open(file_path, "r") do file
        t = read(file["t_array"])
        x = read(file["xgrid"])
        h = read(file["h"])
        b  = read(file["b_exact"])
        return observation_data(t, x, b, h)
    end
end

# Load MCMC chain from file
chain = deserialize("./data/results/chain_mu_s.jls")

# Load the observation data
observation = load_observation_data("./data/toy_measurement/simulation_data.h5")

# Create directory for experiment plots
dir_path = "plots/exp_mu_s"
if !isdir(dir_path)
    mkdir(dir_path)
end
# Plot the results
plot(chain)
savefig("results.png")

bathy_func = x -> swe.gaussian_bathymetry(x, [mean(chain[:peak]), mean(chain[:s])])
reco_bathy = bathy_func(observation.x)
reco_error = abs.(reco_bathy .- observation.b)

# Plot the reconstructed bathymetry
plot(observation.x, observation.b, label="True bathymetry", linecolor=:black)
plot!(observation.x, reco_bathy, label="Reconstructed bathymetry", linecolor=:black, linestyle=:dash)
xlabel!("x [m]")
ylabel!("b [m]")
savefig(joinpath(dir_path,"bathy_reco.pdf"))

# Plot the absolute error
plot(observation.x, reco_error, label="Reconstruction error", linecolor=:black)
xlabel!("x [m]")
ylabel!("Error [m]")
savefig(joinpath(dir_path,"bathy_error.pdf"))