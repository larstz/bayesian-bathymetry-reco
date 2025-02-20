using Pkg
Pkg.activate(".")
using StatsPlots
using Serialization
using PyCall
@pyimport swe_wrapper as swe
using HDF5
using LinearAlgebra
using Statistics
using TOML

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

function bathymetry(x::Array{Float64,1}, μ::Float64, σ²::Float64=1., scale::Float64=0.2)
    return scale * exp.(-1/(σ²+1e-16) .*(x .- μ).^2)
end

exp_name = "experiment_2025-02-18-14-32-18"
exp_dir = "./data/results/$(exp_name)"

config = TOML.parsefile(joinpath(exp_dir,"config_copy.toml"))

# Load MCMC chain from file
chain = deserialize(joinpath(exp_dir,"chain_p.jls"))

# Load the observation data
observation_path = config["observation"]["path"]
sim_name = config["simulation"]["scenario"]*"_"*config["simulation"]["bathymetry"]
observation_file = "simulation_data_$(sim_name).h5"
observation = load_observation_data("./data/toy_measurement/$(observation_file)")

# Create directory for experiment plots
cd(exp_dir)
dir_path = "plots"
if !isdir(dir_path)
    mkdir(dir_path)
end
# Plot the chain of parameters and logp
init_b = config["sampler"]["initial"]
title_init = config["sampler"]["parametrized"] ? "p₀=$(init_b)" : "p₀=exact"
plot(chain[:,1:end-1]; label=["μ" "σ²" "scale"], title="Chain for $(title_init)", xlabel="Iteration", ylabel="Value")
savefig("./plots/chain_iterations.pdf")
plot(chain[:,end]; label="log p(θ)", title="Chain logp for $(title_init)", xlabel="Iteration", ylabel="Value")
savefig("./plots/logp_iterations.pdf")

# Compute the reconstructed bathymetry and error
if config["sampler"]["parametrized"]
    b_reco = bathymetry(observation.x, mean(chain, dims=1)...)
else
    b_reco = mean(chain, dims=1)
end

reco_rel_error = abs.(b_reco .- observation.b)./maximum(observation.b)

# Plot the reconstructed bathymetry
plot(observation.x, observation.b, label="True bathymetry", linecolor=:black)
plot!(observation.x, b_reco, label="Reconstructed bathymetry", linecolor=:black, linestyle=:dash)
xlabel!("x [m]")
ylabel!("b [m]")
savefig(joinpath(dir_path,"bathy_reco.pdf"))

# Plot the absolute error
plot(observation.x, reco_rel_error, label="Reconstruction error; relative L2-error", linecolor=:black)
xlabel!("x [m]")
ylabel!("Error [m]")
savefig(joinpath(dir_path,"reconstruction_rel_error.pdf"))