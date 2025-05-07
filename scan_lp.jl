using Distributed
using TOML
using Serialization
using Distributions
using Dates

addprocs()
println("Added $(nworkers()) workers.")

@everywhere begin
    using BathymetryReco
end

config_file = abspath("config.toml")
toml_config = TOML.parsefile(config_file)
config = load_config(toml_config)
sim_config = config.sim_params
mcmc_config = config.mcmc_params
obs_config = config.obs_settings
io_config = config.io_settings

# Load the data
if obs_config.real_data
    obs_data = load_observation(obs_config.path, sim_config.tstart, sim_config.tinterval)
else
    obs_data, exact_b = load_observation(obs_config.path, obs_config.noise_var; sensor_rate=obs_config.sensor_rate)
end

@everywhere forward_model(params) = simulation(params, $sim_config, $obs_data)

prior_params = [4.5, 0.1, 0.0, 2.0]
likelihood_σ = mcmc_config.likelihood_σ
likelihood_dist = Normal(0, likelihood_σ)
prior_dist = [Normal(prior_params[1:2]...), Uniform(prior_params[3:4]...)]

dist_str = [split("$d", "{")[1] for d in prior_dist]
params_str = [join(prior_params[1+i*2:2+i*2],"_") for i in 0:1]
postfix = "prior_$(dist_str[1])_$(params_str[1])-$(dist_str[2])_$(params_str[2])-likelihood_$(likelihood_σ)"
target_dir = joinpath(io_config.output_dir,
                      "lp_scan_$(postfix)_$(Dates.format(now(), "Y-mm-dd-HH-MM-SS"))")

# add newly calculated information to config
toml_config["sampler"]["likelihood_var"] = likelihood_σ
toml_config["sampler"]["prior"] = "$prior_dist"

pos = Posterior(prior_dist, likelihood_dist)
model = mcmc_model(pos, forward_model, obs_data)

@everywhere begin
    μs = LinRange(1.5,15.0,100)
    σs = LinRange(0.0, 2.0, 50)
    p_grid = [[μ, σ] for μ in μs for σ in σs]
end

println("Grid size: ", length(p_grid))

lps = pmap(p->logjoint(model, p), p_grid)

mkpath(target_dir)
cd(target_dir)
# Save the log-posterior values
serialize("log_posterior_values.jls", lps)
# Save the grid points
serialize("log_posterior_grid_ms.jls", μs)
serialize("log_posterior_grid_ss.jls", σs)
open("./experiment_config.toml", "w") do io
    TOML.print(io, toml_config)
end