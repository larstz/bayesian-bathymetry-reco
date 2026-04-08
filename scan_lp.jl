using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Distributed
using TOML
using Serialization
using Distributions
using Dates
using PDMats
using LinearAlgebra

addprocs(32)
println("Added $(nworkers()) workers.")

@everywhere begin
    using BathymetryReco
end

config_file = abspath("paper_configs/parametrized/experiment_config.toml")
toml_config = TOML.parsefile(config_file)
config = load_config(toml_config)
sim_config = config.sim_params
mcmc_config = config.mcmc_params
obs_config = config.obs_settings
io_config = config.io_settings
sensor_id = obs_config.sensor_id

###############################################################################
# Load the observation data                                                   #
###############################################################################

if obs_config.real_data
    obs_data = load_observation(obs_config.path, sim_config.tstart, sim_config.tinterval,
    sensor_id  = obs_config.sensor_id, noise_var=obs_config.noise_var)
    exp_type = "heat_tests"
    if occursin("mean", obs_config.path)
        exp_type = "heat_tests/mean_tests"
    end
else
    obs_data, exact_b = load_toy_observation(obs_config.path, obs_config.noise_var,
    sensor_rate=obs_config.sensor_rate, sensor_id=obs_config.sensor_id)
    exp_type = "toy_tests"
end

###############################################################################
# Setup the forward model, likelihood, prior and proposal for MCMC sampling   #
###############################################################################

@everywhere forward_model(params) = simulation(params, $sim_config, $obs_data)

# define likelihood distribution
likelihood_σ = mcmc_config.likelihood_σ
if likelihood_σ == 0.0
    flat_signal = forward_model(zeros(mcmc_config.dim))
    likelihood_σ = vec(std(obs_data.H .- flat_signal, dims=1)) # set to std of flat signal residuals
    println("Calculated likelihood std from flat signal residuals: $(likelihood_σ)")
end

println("Using $(likelihood_σ) std for Likelihood distribution.")
likelihood_dist = MvNormal(zeros(size(likelihood_σ)), PDiagMat(likelihood_σ.^2))

# define prior distributions
prior_dist = Vector{Distribution}()
for (i, prior_type) in enumerate(mcmc_config.prior.type)
    prior_param = [mcmc_config.prior.loc[i], mcmc_config.prior.scale[i]]
    if prior_type == "normal"
        push!(prior_dist, Normal(prior_param...))
    elseif prior_type == "uniform"
        push!(prior_dist, Uniform(prior_param...))
    else
        error("Unsupported prior type: $prior_type")
    end
end

# define proposal distribution, not used for likelihood scan but needed to create the model
proposal_kernel = PDMat(Matrix(I, mcmc_config.dim, mcmc_config.dim))
proposal = RandomWalkProposal(mcmc_config.γ, proposal_kernel)

# add newly calculated information to config
toml_config["sampler"]["likelihood_var"] = likelihood_σ

pos = Posterior(prior_dist, likelihood_dist)
model = MCMCModel(pos, forward_model, obs_data, proposal)

target_dir = joinpath("data/results/lp_scan/",
                      "lp_scan_$(Dates.format(now(), "Y-mm-dd-HH-MM-SS"))")

# add newly calculated information to config
toml_config["sampler"]["likelihood_var"] = likelihood_σ

pos = Posterior(prior_dist, likelihood_dist)
model = MCMCModel(pos, forward_model, obs_data, Normal(0,1))

###############################################################################
# Run likelihood scan                                                #
###############################################################################

@everywhere begin
    μs = LinRange(1.5,12.5,23)
    σs = LinRange(0.01, 0.5, 50)
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