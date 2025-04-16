using Pkg
Pkg.activate(".")
Pkg.instantiate()
using Distributed

addprocs(8)

using Dates
using TOML
using Serialization
using Plots

@everywhere begin
    using Distributions
    using BathymetryReco
end

# Load the configuration
println("#############################\nRead in config file" )
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
    obs_data, exact_b = load_observation(obs_config.path, obs_config.noise_var)
end

# create plot of the observation signal
ps = plot(;title="Observation signal", xlabel="time [s]", ylabel="Water surface displacement [m]")
plot!(ps, obs_data.t, obs_data.H; label=reshape(["Sensor $i" for i in 2:4], 1,3))
exp_name = splitpath(obs_config.path)[end]

store_exp = io_config.save
target_dir = joinpath(io_config.output_dir,
                      "$(exp_name)_$(Dates.format(now(), "Y-mm-dd-HH-MM-SS"))")

if store_exp
    mkpath(target_dir)
    plot_path = joinpath(target_dir,"plots")
    mkpath(plot_path)
    savefig(ps, joinpath(plot_path,"observation_signal.pdf"))
end

@everywhere forward_model(params) = simulation(params, $sim_config, $obs_data)

likelihood_σ = mcmc_config.likelihood_σ
if likelihood_σ == 0.0
    likelihood_σ = maximum(obs_data.noise_std) # replace later by individual σ for each
end

println("Using $(likelihood_σ) std for Likelihood distribution.")
likelihood_dist = Normal(0, likelihood_σ)
prior_dist = [Normal(4.5,1), Uniform(0, 2)]

# add newly calculated information to config
toml_config["sampler"]["likelihood_var"] = likelihood_σ
toml_config["sampler"]["prior"] = "$prior_dist"

pos = Posterior(prior_dist, likelihood_dist)
model = mcmc_model(pos, forward_model, obs_data)

init_θ = mcmc_config.initial_θ
if isempty(mcmc_config.initial_θ)
    init_θ = [vec(vcat(rand.(prior_dist,1)...)) for i in 1:mcmc_config.n_chains]
end
println("#############################")
println(size(init_θ))
println(typeof(init_θ))
for i in 1:mcmc_config.n_chains
    println(init_θ[i])
end
println("#############################")
println("Start $(mcmc_config.n_chains) chains with $(mcmc_config.n) samples: \n#############################" )
chain = pmap(1:mcmc_config.n_chains) do i
    sample_chain(model, mcmc_config, init_θ[i])
end
println("Chains finished \n#############################" )
if store_exp
    mkpath(target_dir)
    cd(target_dir)

    # store the configuration file for reproducibility
    open("./experiment_config.toml", "w") do io
        TOML.print(io, toml_config)
    end

    pc = plot(;title="Chains", xlabel="Iteration", ylabel="Value")
    plp = plot(;title="Chain log p(θ)", xlabel="Iteration", ylabel="Value")
    # Serialize the chain
    for (i, initial_θ) in enumerate(mcmc_config.initial_θ)
        serialize("chain_$i.jls", chain[i])

        # Plot the chain
        plot!(pc,chain[i][:,1:end-1]; label=["μ_$i" "σ²_$i"]) # sampled parameters
        plot!(plp, chain[i][:,end]; label="$i: log p(θ)") # log p of sample
    end
    savefig(pc, "./plots/chain.pdf")
    savefig(plp, "./plots/logp.pdf")
end