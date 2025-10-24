using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Dates
using TOML
using Serialization
using Plots

using Distributions
using PDMats
using BathymetryReco
using ProgressMeter


ENV["GKSwstype"]="nul"

# Load the configuration
println("#############################\nRead in config file" )
if isempty(ARGS)
    config_file = abspath("config.toml")
else
    config_file = abspath(ARGS[1])
end

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
    obs_data, exact_b = load_observation(obs_config.path, obs_config.noise_var, sensor_rate=obs_config.sensor_rate)
end

# create plot of the observation signal
ps = plot(;title="Observation signal", xlabel="time [s]", ylabel="Water surface height [m]")
plot!(ps, obs_data.t, obs_data.H; label=reshape(["Sensor $i" for i in 2:4], 1,3))
exp_name = splitpath(obs_config.path)[end]

store_exp = io_config.save
target_dir = joinpath(io_config.output_dir,
                      "$(exp_name)_$(Dates.format(now(), "Y-mm-dd-HH-MM-SS"))")

if store_exp
    mkpath(target_dir)
    plot_path = joinpath(target_dir,"plots")
    mkpath(plot_path)
    savefig(ps, joinpath(plot_path,"observation_signal.png"))
end

solver = swe_solver(sim_config)

forward_model(params) = simulation(params, solver, obs_data)

likelihood_σ = mcmc_config.likelihood_σ
if likelihood_σ == 0.0
    max_signal = [0.0015, 0.003, 0.003]
    #likelihood_σ = maximum(obs_data.noise_std' ./ max_signal) # replace later by individual σ for each
    likelihood_σ = vec(obs_data.noise_std) ./ max_signal
end

println("Using $(likelihood_σ) std for Likelihood distribution.")
#likelihood_dist = Normal(0, likelihood_σ)
likelihood_dist = MvNormal(zeros(size(likelihood_σ)), PDiagMat(likelihood_σ.^2))
xs = collect(range(sim_config.xbounds[1], sim_config.xbounds[2], length=mcmc_config.dim))
s = 0.005.*exp.(-1/(xs[3]-xs[1]).^2 .*(xs.-xs').^2) # smooth prior
# prior_dist = [Cauchy(0., 0.01) for i in 1:length(sim_config.nx)] # sparse prior
prior_dist = [Cauchy(0., 0.01), MvNormal(zeros(mcmc_config.dim), PDMat(s))] # sqexp prior

# add newly calculated information to config
println("Using likelihood std $(likelihood_σ) for Likelihood distribution.")
toml_config["sampler"]["likelihood_var"] = likelihood_σ
toml_config["sampler"]["prior"] = "$prior_dist"


pos = Posterior(prior_dist, likelihood_dist)
model = mcmc_model(pos, forward_model, obs_data)

init_θ = mcmc_config.initial_θ

if isempty(init_θ)
    #init_θ = [vec(vcat(rand.(prior_dist,1)...)) for i in 1:mcmc_config.n_chains]
    #init_θ = [exp_bathymetry(solver.domain.x) for i in 1:mcmc_config.n_chains]
    init_θ = [bathymetry(xs, [4.5,0.05]) for i in 1:mcmc_config.n_chains]
    init_θ[1] .= 0.0 #exp_bathymetry(xs) # set first chain to correct bathymetry
    toml_config["sampler"]["init"] = init_θ
    inip = plot(xs, init_θ[1])
    savefig(inip, joinpath(plot_path, "initial_parameters.png"))
end
println("#############################")
println(size(init_θ))
println(typeof(init_θ))
for i in 1:mcmc_config.n_chains
    println(init_θ[i])
end
println("#############################")
println("Start $(mcmc_config.n_chains) chains with $(mcmc_config.n) samples: \n#############################" )

chains = []
for i in 1:mcmc_config.n_chains
    chain = sample_chain(model, mcmc_config, init_θ[i], verbose=true, logging=Progress(mcmc_config.n))
    push!(chains, chain)
end
println("Chains finished \n#############################" )
println(size(chains))
println
if store_exp
    mkpath(target_dir)
    cd(target_dir)

    # store the configuration file for reproducibility
    open("./experiment_config.toml", "w") do io
        TOML.print(io, toml_config)
    end

    pc = plot(;title="Chains", xlabel="Iteration", ylabel="Value", legend=:outerright)
    plp = plot(;title="Chain log p(θ)", xlabel="Iteration", ylabel="Value", legend=:outerright)
    pla = plot(;title="Chain acceptance rate α", xlabel="Iteration", ylabel="Value", legend=:outerright)
    pll = plot(;title="Chain log likelihood", xlabel="Iteration", ylabel="Value", legend=:outerright)
    plprior = plot(;title="Chain log prior", xlabel="Iteration", ylabel="Value", legend=:outerright)
    # Serialize the chain
    for (i, initial_θ) in enumerate(init_θ)
        serialize("chain_$i.jls", chains[i])

        # Plot the chain
        plot!(pc,chains[i][:,1:end-4]; label="") # sampled parameters
        plot!(plp, chains[i][:,end-3]; label="$i: log p(θ)") # log p
        plot!(pll, chains[i][:,end-2]; label="$i: log likelihood") # log likelihood
        plot!(plprior, chains[i][:,end-1]; label="$i: log prior") # log prior
        plot!(pla, chains[i][:,end]; label="$i: α") # acceptance rate
    end
    savefig(pc, "./plots/chain.png")
    savefig(plp, "./plots/logp.png")
    savefig(pll, "./plots/loglikelihood.png")
    savefig(plprior, "./plots/logprior.png")
    savefig(pla, "./plots/acceptance_rate.png")
end