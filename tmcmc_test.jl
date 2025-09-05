using UncertaintyQuantification

using TOML
using Distributions
using PDMats
using BathymetryReco
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

solver = swe_solver(sim_config)

forward_model_df = df -> [simulation([params.μ, params.σ², params.s], solver, obs_data) .- obs_data.H for params in eachrow(df)]

rv_params = [RandomVariable(Uniform(sim_config.xbounds[1], sim_config.xbounds[2]), :μ),
             RandomVariable(Uniform(0.01, 0.1), :σ²),
             RandomVariable(Uniform(0.001, 0.3), :s)]
prior = [Uniform(sim_config.xbounds[1], sim_config.xbounds[2]),
         Uniform(0.01, 0.1),
         Uniform(0.001, 0.3)]

logprior_df = df -> logpdf.(prior[1], df.μ) .+ logpdf.(prior[2], df.σ²) .+ logpdf.(prior[3], df.s)
# use model as swe solver with random variables
model = Model(forward_model_df, :swe)


likelihood_σ = mcmc_config.likelihood_σ
if likelihood_σ == 0.0
    likelihood_σ = maximum(obs_data.noise_std) # replace later by individual σ for each
end

println("Using $(likelihood_σ) std for Likelihood distribution.")
likelihood_dist = Normal(0, likelihood_σ)

loglikelihood_df = df -> [sum(logpdf.(likelihood_dist, d.swe)) for d in eachrow(df)]

burnin = 1
n_samples = 100
tmcmc = TransitionalMarkovChainMonteCarlo(rv_params, n_samples, burnin)
@time tmcmc_samples, S = bayesianupdating(logprior_df, loglikelihood_df, [model], tmcmc)

xs = collect(range(sim_config.xbounds[1], sim_config.xbounds[2], length=sim_config.nx))
s = 0.005.*exp.(-1/(xs[3]-xs[1]).^2 .*(xs.-xs').^2) # smooth prior
# prior_dist = [Cauchy(0., 0.01) for i in 1:length(sim_config.nx)] # sparse prior
prior_dist = [Cauchy(0., 0.01), MvNormal(zeros(sim_config.nx), PDMat(s))] # sqexp prior

# add newly calculated information to config
toml_config["sampler"]["likelihood_var"] = likelihood_σ
toml_config["sampler"]["prior"] = "$prior_dist"


pos = Posterior(prior_dist, likelihood_dist)
model = mcmc_model(pos, forward_model, obs_data)

init_θ = mcmc_config.initial_θ

if isempty(init_θ)
    xg = collect(range(sim_config.xbounds[1], sim_config.xbounds[2], length=sim_config.nx))
    #init_θ = [vec(vcat(rand.(prior_dist,1)...)) for i in 1:mcmc_config.n_chains]
    #init_θ = [exp_bathymetry(solver.domain.x) for i in 1:mcmc_config.n_chains]
    init_θ = [bathymetry(xg, [4.5,0.05]) for i in 1:mcmc_config.n_chains]
    init_θ[1] .= exp_bathymetry(xg)#0.0 # set first chain to correct bathymetry
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

chains = [] # +2 for logp and acceptance rate
for i in 1:mcmc_config.n_chains
    chain = sample_chain(model, mcmc_config, init_θ[i], verbose=true, logging=Progress(mcmc_config.n))
    push!(chains, chain)
end
println("Chains finished \n#############################" )
