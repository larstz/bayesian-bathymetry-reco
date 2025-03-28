using Pkg
Pkg.activate(".")
Pkg.instantiate()
using Distributed

addprocs(4)

using Dates
using TOML
using Serialization
using Plots

@everywhere begin
    using Distributions
    using BathymetryReco
end

# Load the configuration
config_file = abspath("config.toml")
config = load_config(config_file)
sim_config = config.sim_params
mcmc_config = config.mcmc_params
obs_config = config.obs_settings
io_config = config.io_settings

# Load the data
obs_data, exact_b = load_observation(obs_config.path,
                                                obs_config.noise_var)

# create plot of the observation signal
ps = plot(;title="Observation signal", xlabel="t [s]", ylabel="z [m]")
plot!(ps, obs_data.t, obs_data.H; label=reshape(["Sensor $i" for i in 2:4], 1,3))
exp_name = splitpath(obs_config.path)[end]

store_exp = io_config.save
target_dir = joinpath(io_config.output_dir,
                      "$(exp_name)_$(Dates.format(now(), "Y-mm-dd-HH-MM-SS"))")

@everywhere forward_model(params) = simulation(params, $sim_config, $obs_data)

likelihood_dist = Normal(0, mcmc_config.likelihood_σ)
prior_dist = product_distribution([Uniform(sim_config.xbounds...), Uniform(0, 2)])

pos = Posterior(prior_dist, likelihood_dist)
model = mcmc_model(pos, forward_model, obs_data)

chain = pmap(x->sample_chain(model, mcmc_config, x), mcmc_config.initial_θ)

if store_exp
    mkpath(target_dir)
    cd(target_dir)

    # store the configuration file for reproducibility
    cp(config_file, "./config_copy.toml")

    mkpath("./plots")
    pc = plot(;title="Chains", xlabel="Iteration", ylabel="Value")
    plp = plot(;title="Chain log", xlabel="Iteration", ylabel="Value")
    # Serialize the chain
    for (i, initial_θ) in enumerate(mcmc_config.initial_θ)
        serialize("chain_$i.jls", chain[i])

        # Plot the chain
        plot!(pc,chain[i][:,1:end-1]; label=["μ_$i" "σ²_$i"])
        plot!(plp, chain[i][:,end]; label="$i: log p(θ)")
    end
    savefig(ps, "observation_signal.pdf")
    savefig(pc, "./plots/chain.pdf")
    savefig(plp, "./plots/logp.pdf")
end