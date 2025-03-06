using Pkg
Pkg.activate(".")
Pkg.instantiate()
using Distributed
using Distributions
using Dates
using TOML
using BathymetryReco

# Load the configuration
config = load_config("config.toml")
sim_config = config.sim_params
mcmc_config = config.mcmc_params
obs_config = config.obs_settings
io_config = config.io_settings

# Load the data
obs_data, exact_b = load_observation(obs_config.path,
                                                obs_config.noise_var)

exp_name = splitpath(obs_config.path)[end]

store_exp = io_config.save
target_dir = joinpath(io_config.output_dir,
                      "$(exp_name)_$(Dates.format(now(), "Y-mm-dd-HH-MM-SS"))")

forward_model(params) = simulation(params, sim_config, obs_data)
forward_model([4.,0.1])
likelihood_dist = Normal(0, mcmc_config.likelihood_σ)
prior_dist = product_distribution([Uniform(sim_config.xbounds...), Uniform(0, 2)])

pos = Posterior(prior_dist, likelihood_dist)
model = mcmc_model(pos, forward_model, obs_data)

chain = sample_chain(model, mcmc_config)

if store_exp
    mkpath(target_dir)
    cd(target_dir)

    # Serialize the chain
    serialize("chain_p.jls", chain)

    # store the configuration file for reproducibility
    open("config_copy.toml", "w") do io
        TOML.print(io, config)
    end
    # Plot the chain
    title_init = config["sampler"]["parametrized"] ? "p₀=$(init_b)" : "p₀=exact"
    plot(chain[:,1:end-1]; label=["μ" "σ²" "scale"], title="Chain for $(title_init)", xlabel="Iteration", ylabel="Value")
    mkpath("./plots")
    savefig("./plots/chain.pdf")
    plot(chain[:,end]; label="log p(θ)", title="Chain logp for $(title_init)", xlabel="Iteration", ylabel="Value")
    savefig("./plots/logp.pdf")
end