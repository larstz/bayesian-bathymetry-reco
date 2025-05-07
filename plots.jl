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
    x = Vector(LinRange(sim_config.xbounds[1], sim_config.xbounds[2], sim_config.nx))
    exact_b = exp_bathymetry(x)
else
    obs_data, exact_b = load_observation(obs_config.path, obs_config.noise_var)
    x = obs_data.sim_x
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

# everything should be loaded now create the plots
plot_path = joinpath(experiment, "plots")
mkpath.(joinpath.(plot_path, ["pdfs", "pngs"]))

println("Plot the MCMC samples")
# Plot the chains
for i in 1:stored_vals
    param = param_names[i]
    param_per_chain = getindex.(samples, :, i)
    pc = plot(;title="Chains for param $(param)", xlabel="Iteration", ylabel="Value")
    plot!(pc, param_per_chain, label=permutedims("$(param)_".*string.(1:n_chains)))
    savefig(pc, joinpath(plot_path, "pngs", "chain_val_$(i).png"))
    savefig(pc, joinpath(plot_path, "pdfs", "chain_val_$(i).pdf"))
end

println("Plot the bathymetry samples and sensor simulations per bathymetry")
# Plot Bathymetries and sensor simulations
for (idx, chain) in enumerate(samples)
    # Plot the bathymetry
    bathys = zeros(size(chain)[1], length(exact_b))
    for (i, sample) in enumerate(eachrow(chain))
        bathys[i, :] = bathymetry(x, sample[1:end-1])
    end
    bathy_mean = vec(mean(bathys, dims=1))
    mean_params = vec(mean(chain[:, 1:end-1], dims=1))
    rel_l2_error_mean = round.(sqrt(sum((bathy_mean .- exact_b).^2)) / sqrt(sum((exact_b).^2)), digits=4)
    rel_l2_error = round.(sqrt(sum((bathymetry(x, mean_params) .- exact_b).^2)) ./ sqrt(sum((exact_b).^2)), digits=4)
    pb = plot(x, exact_b; c=:black, title="Bathymetry", xlabel="x [m]", ylabel="b [m]", label="exact")
    plot!(pb, x, bathys'; label=permutedims(vcat(["Samples"], repeat([""], size(bathys)[1]))), alpha=0.1, lw=0.25, color=:gray)
    plot!(pb, x, bathy_mean; c=:red, label="sample mean, ε=$(rel_l2_error_mean)")
    plot!(pb, x, bathymetry(x, mean_params); c=:blue, label="mean params, ε=$(rel_l2_error)")
    # to display error use bars from quantiles
    savefig(pb, joinpath(plot_path, "pngs", "bathy_chain_$(idx).png"))
    savefig(pb, joinpath(plot_path, "pdfs", "bathy_chain_$(idx).pdf"))

    # Plot the sensor simulation
    sim_chain = simulation(mean_params, sim_config, obs_data)
    rel_l2_sim_error = round.(sqrt.(sum((sim_chain .- obs_data.H).^2, dims=1)) ./ sqrt.(sum((obs_data.H).^2, dims=1)), digits=4)
    for i in 2:4
        psim = plot(obs_data.t, obs_data.H[:,i-1]; title="Sensor $i, ε=$(rel_l2_sim_error[i-1])", label="measurement", xlabel="t [s]", ylabel="H [m]", linestyle=:dash)
        plot!(psim, obs_data.t, sim_chain[:,i-1]; label="simulation ", linestyle=:dot, linewidth=2)
        savefig(psim, joinpath(plot_path, "pngs", "sim_chain_$(idx)_sensor_$(i).png"))
        savefig(psim, joinpath(plot_path, "pdfs", "sim_chain_$(idx)_sensor_$(i).pdf"))
    end

end

