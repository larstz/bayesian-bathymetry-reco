using Pkg
Pkg.activate(".")

using BathymetryReco
using DataFrames
using Statistics
using Serialization
using Dates
using LaTeXStrings
using MCMCChains
using MC


date_pattern = r"(\d{4}-\d{2}-\d{2})"

experiment = ARGS[1]
exp_date = Date(match(date_pattern, experiment).match, DateFormat("Y-mm-dd"))
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
    obs_data, exact_b = load_observation(obs_config.path, obs_config.noise_var, sensor_rate=obs_config.sensor_rate)
    x = obs_data.sim_x
end

# Load the MCMC samples
files = readdir(experiment)
chains = filter(x -> occursin(r"chain_[0-9]+.jls", x), files)
n_chains = length(chains)
samples = [deserialize(joinpath(experiment, file)) for file in chains]
burn_in = 250

# sampled 2σ² instead of σ², so we need to convert it, adjusted after 2025-06-25
if exp_date < Date(2025,6,25)
    setindex!.(samples, 0.5 .* getindex.(samples, :, 2), :,2)
end

sample_mean = mean.(samples, dims=1)
samples_mat = hcat(samples...)
chain_tensor = cat(samples_mat..., dims=3)

stored_vals = Int64(size(samples_mat, 2)/n_chains)
param_names = ["mu", "s2", "lp", "ar"]

mcmc_chain = Chains(chain_tensor, param_names, Dict(:internal => [:lp, :ar]))
df_chains = DataFrame(mcmc_chains)
