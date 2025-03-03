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
    t_sim::Array{Float64}
    t_reco::Array{Float64}
    x::Array{Float64}
    b::Array{Float64}
    h::Array{Float64}
    H_sensor_sim::Array{Float64}
    H_sensor_reco::Array{Float64}
    sensor::Array{Float64}
end

function load_observation_data(file_path::String)
    h5open(file_path, "r") do file
        dt = attrs(file)["dt"]
        tinterval = attrs(file)["T_N"]
        t = collect(0:dt:tinterval)
        t_reco = collect(0:0.001:10)
        tid = findall(x->x in t_reco, t)
        x = read(file["xgrid"])
        h = read(file["h"])
        b  = read(file["b_exact"])
        H_sensor = read(file["H_sensor"])
        H_sensor_reco = H_sensor[tid, :]
        sensor = [3.5, 5.5, 7.5]
        return observation_data(t, t_reco, x, b, h, H_sensor, H_sensor_reco, sensor)
    end
end

function bathymetry(x::Array{Float64,1}, μ::Float64, σ²::Float64=1., scale::Float64=0.2)
    return scale * exp.(-1/(σ²+1e-16) .*(x .- μ).^2)
end

function plot_bathy(x::Array{Float64,1}, b::Function, observation::observation_data)
    b_reco = b(x)
    if size(b_reco) == size(observation.b)
        reco_rel_error = abs.((b_reco .- observation.b)./observation.b)
    else
        reco_rel_error = 0
    end

    # Plot the reconstructed bathymetry
    p1 = plot(observation.x, observation.b, label="True bathymetry", linecolor=:black)
    plot!(p1, x, b_reco, label="Reconstructed bathymetry", linecolor=:black, linestyle=:dash)
    xlabel!(p1, "x [m]")
    ylabel!(p1, "b [m]")
    # Plot the absolute error
    p2 = plot(x, 100*reco_rel_error, label="Reconstruction error", linecolor=:black)
    xlabel!(p2, "x [m]")
    ylabel!(p2, "Error [%]")
    return p1, p2
end

function sensor_plots(observation::observation_data, sim_observations::Array{Float64,2})

    if size(sim_observations) == size(observation.H_sensor_sim)
       t = observation.t_sim
       H_obs = observation.H_sensor_sim
    elseif size(sim_observations) == size(observation.H_sensor_reco)
        t = observation.t_reco
        H_obs = observation.H_sensor_reco
    else
        error("Size of sensor data does not match")
    end
    # Plot the sensor data
    p1 = plot(t, H_obs, label=reshape(["Sensor $i" for i in 2:4], 1, 3))
    plot!(p1, t, sim_observations, label=reshape(["Reco Sensor $i" for i in 2:4], 1, 3), linestyle=:dash)
    xlabel!(p1, "t [s]")
    ylabel!(p1, "z [m]")
    title!(p1, "Sensor positions")


    # Plot sensor error
    sensor_error = abs.((sim_observations.- H_obs)./H_obs)
    p2 = plot(t, 100*sensor_error, label=reshape(["Sensor $i" for i in 2:4], 1, 3))
    xlabel!(p2, "t [s]")
    ylabel!(p2, "Error [%]")
    title!(p2, "Sensor error")
    return p1, p2
end

exp_name = "waterchannel_exact_bathy_2025-02-28-13-08-45"
exp_dir = "./data/results/$(exp_name)"

config_reco = TOML.parsefile(joinpath(exp_dir,"config_copy.toml"))

# Load MCMC chain from file
chain = deserialize(joinpath(exp_dir,"chain_p.jls"))

# Load the observation data
observation_path = config_reco["observation"]["path"]
sim_name = config_reco["simulation"]["scenario"]*"_"*config_reco["simulation"]["bathymetry"]
observation_file = "jl_simulation_data.h5"
observation = load_observation_data(joinpath(observation_path, sim_name, observation_file))
config_sim = TOML.parsefile(joinpath(observation_path, sim_name, "simulation_config.toml"))
# Create directory for experiment plots
cd(exp_dir)
dir_path = "plots"
if !isdir(dir_path)
    mkdir(dir_path)
end

# # Plot the chain of parameters and logp
# init_b = config["sampler"]["initial"]
# title_init = config["sampler"]["parametrized"] ? "p₀=$(init_b)" : "p₀=exact"
# plot(chain[:,1:end-1]; label=["μ" "σ²" "scale"], title="Chain for $(title_init)", xlabel="Iteration", ylabel="Value")
# savefig("./plots/chain_iterations.pdf")
# plot(chain[:,end]; label="log p(θ)", title="Chain logp for $(title_init)", xlabel="Iteration", ylabel="Value")
# savefig("./plots/logp_iterations.pdf")

# Compute the reconstructed bathymetry and error
if config_reco["sampler"]["parametrized"]
    b_reco(x) = bathymetry(x, mean(chain[500:end,1:end-1], dims=1)...)
else
    b_reco = mean(chain, dims=1)
end

# Run simulation with reco setting
params_reco = config_reco["simulation"]
solver_reco = swe.SWESolver(params_reco["xbounds"], params_reco["timestep"],
                            params_reco["nx"], params_reco["tinterval"], g=params_reco["g"],
                            kappa=params_reco["kappa"], dealias=params_reco["dealias"],
                            tstart=params_reco["tstart"],
                            problemtype=params_reco["scenario"],);
x_reco = solver_reco.domain.x
b_reco_reco = b_reco(x_reco)
reco_observations, _, _, _ = solver_reco.solve(b_reco_reco, sensor_pos=observation.sensor)

bathy_plots_reco = plot_bathy(x_reco, b_reco, observation)
savefig(bathy_plots_reco[1], joinpath(dir_path,"reco_bathy_reco_w_500_burnin.pdf"))
savefig(bathy_plots_reco[2], joinpath(dir_path,"reco_reconstruction_rel_error_w_500_burnin.pdf"))

reco_sensor_plots = sensor_plots(observation, reco_observations)
savefig(reco_sensor_plots[1], joinpath(dir_path,"reco_sensor_positions_plots.pdf"))
savefig(reco_sensor_plots[2], joinpath(dir_path,"reco_sensor_error_plots.pdf"))

# Run simulation with toy measurement setting
params_sim = config_sim["simulation"]
solver_sim = swe.SWESolver(params_sim["xbounds"], params_sim["timestep"],
                            params_sim["nx"], params_sim["tinterval"], g=params_sim["g"],
                            kappa=params_sim["kappa"], dealias=params_sim["dealias"],
                            tstart=params_sim["tstart"],
                            problemtype=params_sim["scenario"],);
x_sim = solver_sim.domain.x
b_reco_sim = b_reco(x_sim)
sim_observations, _, _, _ = solver_sim.solve(b_reco_sim, sensor_pos=observation.sensor)

bathy_plots_sim = plot_bathy(x_sim, b_reco, observation)
savefig(bathy_plots_sim[1], joinpath(dir_path,"sim_bathy_reco_w_500_burnin.pdf"))
savefig(bathy_plots_sim[2], joinpath(dir_path,"sim_reconstruction_rel_error_w_500_burnin.pdf"))

sim_sensor_plots = sensor_plots(observation, sim_observations)
savefig(sim_sensor_plots[1], joinpath(dir_path,"sim_sensor_positions_plots.pdf"))
savefig(sim_sensor_plots[2], joinpath(dir_path,"sim_sensor_error_plots.pdf"))
