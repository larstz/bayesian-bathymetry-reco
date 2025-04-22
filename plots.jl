using BathymetryReco
using Plots
using DataFrames
using Statistics
using Serialization

experiment = ARGS[1]
println("Creating plots for experiment: ", experiment)
config = load_config(joinpath(experiment, "experiment_config.toml"))
sim_config = config.sim_params
mcmc_config = config.mcmc_params
obs_config = config.obs_settings
io_config = config.io_settings

# Load the data
if obs_config.real_data
    obs_data = load_observation(obs_config.path, sim_config.tstart, sim_config.tinterval)
else
    obs_data, exact_b = load_observation(obs_config.path, obs_config.noise_var)
end

# Load the MCMC samples
files = readdir(experiment)
chains = filter(x -> occursin(r"chain_[0-9]+.jls", x), files)
n_chains = length(chains)
samples = [deserialize(joinpath(experiment, file)) for file in chains]
sample_mean = mean.(samples, dims=1)
samples_mat = hcat(samples...)
stored_vals = Int64(size(samples_mat, 2)/n_chains)
param_names = ["μ", "σ", "lp"]

function chains2df(chains)
    # Convert the chains to a DataFrame
    df = DataFrame()
    i = 1
    for chain in chains
        n_params = size(chain, 2) - 1
        nms = vcat([Symbol("param_$(j)_c_$(i)") for j in 1:n_params]..., Symbol("lp_c_$(i)"))
        chain_df = DataFrame(chain, nms)
        df = hcat(df, chain_df)# leftjoin!(df, chain_df)
        i += 1
    end
    return df
end

println("Plot the MCMC samples")
# Plot the chains
for i in 1:stored_vals
    param = param_names[i]
    param_per_chain = getindex.(samples, :, i)
    pc = plot(;title="Chains for param $(param)", xlabel="Iteration", ylabel="Value")
    plot!(pc, param_per_chain, label=permutedims("$(param)_".*string.(1:n_chains)))
    savefig(joinpath(experiment, "plots", "chain_val_$(i).png"))
end

println("Plot the bathymetry samples and sensor simulations per bathymetry")
# Plot Bathymetries and sensor simulations
for (idx, chain) in enumerate(samples)
    # Plot the bathymetry
    bathys = zeros(size(chain)[1], length(exact_b))
    x = Vector(LinRange(sim_config.xbounds[1], sim_config.xbounds[2], length(exact_b)))
    for (i, sample) in enumerate(eachrow(chain))
        bathys[i, :] = bathymetry(x, sample[1:end-1])
    end
    bathy_mean = mean(bathys, dims=1)
    rel_l2_error = sqrt(sum((bathy_mean .- exact_b).^2)) / sqrt(sum((exact_b).^2))
    pb = plot(x, exact_b; c=:black, title="Bathymetry; rel_l2_error=$(rel_l2_error)", xlabel="x [m]", ylabel="y [m]", label="True b")
    plot!(pb, x, bathy_mean'; c=:red, label="sample mean b")
    plot!(pb, x, bathys'; label="Samples", alpha=0.1, lw=0.25)
    # to display error use bars from quantiles
    savefig(joinpath(experiment, "plots", "bathy_chain_$(idx).png"))

    # Plot the sensor simulation
    mean_params = mean(chain[1:end-1], dims=1)
    sim_chain = simulation(mean_params, sim_config, obs_data)
    rel_l2_sim_error = sqrt.(sum((sim_chain .- obs_data.H).^2, dims=1)) ./ sqrt.(sum((obs_data.H).^2, dims=1))
    psim = plot(obs_data.t, obs_data.H; title="Sensor simulation, rel_l2_error=$(rel_l2_sim_error...)", label=reshape(["Sensor $i" for i in 2:4], 1, 3))
    plot!(psim, obs_data.t, sim_chain; label=reshape(["Sim Sensor $i" for i in 2:4], 1, 3))
    savefig(joinpath(experiment, "plots", "sim_chain_$(idx).png"))
end

