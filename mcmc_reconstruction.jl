using Pkg
Pkg.activate(".")
#Pkg.instantiate()
using Distributed

addprocs(7)

using Dates
using TOML
using Serialization
using Plots
using PDMats
using LinearAlgebra

@everywhere begin
    using Distributions
    using BathymetryReco
end

ENV["GKSwstype"]="nul"

# Load the configuration
println("#############################\nRead in config file" )
if isempty(ARGS)
    config_file = abspath("./width_test_configs/config_1.toml")
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

@everywhere forward_model(params) = simulation(params, $sim_config, $obs_data)

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
proposal_dist = MvNormal(zeros(mcmc_config.dim),mcmc_config.γ.^2 .* PDiagMat(ones(mcmc_config.dim)))

# add newly calculated information to config
toml_config["sampler"]["likelihood_var"] = likelihood_σ

pos = Posterior(prior_dist, likelihood_dist)
model = MCMCModel(pos, forward_model, obs_data, proposal_dist)

init_θ = mcmc_config.initial_θ

if isempty(init_θ)
    xg = collect(range(sim_config.xbounds[1], sim_config.xbounds[2], length=sim_config.nx))
    #init_θ = [vec(vcat(rand.(prior_dist,1)...)) for i in 1:mcmc_config.n_chains]
    #init_θ = [exp_bathymetry(solver.domain.x) for i in 1:mcmc_config.n_chains]
    init_θ = [zeros(mcmc_config.dim)]#[bathymetry(xg, [4.5,0.05]) for i in 1:mcmc_config.n_chains]
    toml_config["sampler"]["init"] = init_θ
    inip = plot(xg, init_θ[1])
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
chain = pmap(1:mcmc_config.n_chains) do i
    sample_chain(model, mcmc_config, init_θ[i])
end
println("Chains finished \n#############################" )
println(size(chain))
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
        serialize("chain_$i.jls", chain[i])

        # Plot the chain
        plot!(pc,chain[i][:,1:end-4]; label="") # sampled parameters
        plot!(plp, chain[i][:,end-3]; label="$i: log p(θ)") # log p
        plot!(pll, chain[i][:,end-2]; label="$i: log likelihood") # log likelihood
        plot!(plprior, chain[i][:,end-1]; label="$i: log prior") # log prior
        plot!(pla, chain[i][:,end]; label="$i: α") # acceptance rate
    end
    savefig(pc, "./plots/chain.png")
    savefig(plp, "./plots/logp.png")
    savefig(pll, "./plots/loglikelihood.png")
    savefig(plprior, "./plots/logprior.png")
    savefig(pla, "./plots/acceptance_rate.png")
end
