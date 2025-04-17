using BathymetryReco
using Plots
using DataFrames
using Statistics
using Serialization

experiment = ARGS[1]

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
chains = filter(x -> occursin(r"chain_[1-9]+.jls", x), files)
n_chains = length(chains)
samples = [deserialize(joinpath(experiment, file)) for file in chains]
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

# Plot the chains
for i in 1:stored_vals
    param = param_names[i]
    param_per_chain = getindex.(samples, :, i)
    p = plot(;title="Chains for param $(param)", xlabel="Iteration", ylabel="Value")
    plot!(p, param_per_chain, label=permutedims("$(param)_".*string.(1:n_chains)))
    savefig(joinpath(experiment, "plots", "chain_val_$(i).png"))
end

# Plot Bathymetries
for chain in samples
    # Plot the bathymetry
    bathy_params_m = mean(chain[:, 1:end-1], dims=1)

    p = plot(;title="Bathymetry", xlabel="X", ylabel="Y")
end
# Plot sensors for sampled bathymetry