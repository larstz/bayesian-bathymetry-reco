###############################################################################
# MCMC reconstruction of bathymetry from water surface height observations    #
# This script runs multiple MCMC chains in serial and stores the results      #
# along with diagnostic plots.                                                #
#                                                                             #
#   Author: Lars Stietz                                                       #
###############################################################################

using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Dates
using TOML
using Serialization
using Plots

using Distributions
using PDMats
using LinearAlgebra
using BathymetryReco
using ProgressMeter

using FFTW
using StatsBase
using LsqFit
using Random

Random.seed!(1910)

ENV["GKSwstype"]="nul"

###############################################################################
# Load the configuration                                                      #
###############################################################################

println("#############################\nRead in config file" )
if isempty(ARGS)
    config_file = abspath("./paper_configs/discretized/mean_heat_config.toml")
else
    config_file = abspath(ARGS[1])
end

toml_config = TOML.parsefile(config_file) # load toml to modify later
config = load_config(toml_config)
sim_config = config.sim_params
mcmc_config = config.mcmc_params
obs_config = config.obs_settings
prior_settings = mcmc_config.prior
proposal_settings = mcmc_config.proposal
io_config = config.io_settings



println("##############################\nLoad experiment data")

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

# Set up directory for storing results
store_exp = io_config.save
exp_name = split(splitpath(obs_config.path)[end], ".")[1]
# Directory structure for storing results experiment/sensors/prior/proposal/stepsize/timestamp_expname
target_dir = joinpath(io_config.output_dir,
                      exp_type,
                      "sensor-"*join(obs_config.sensor_id, "-"),
                      "prior-"*join(prior_settings.type,"-"),
                      "proposal-"*proposal_settings.type * "-" * proposal_settings.kernel,
                      "stepsize-"*join(string.(mcmc_config.γ),"-"),
                      "$(Dates.format(now(), "Y-mm-dd-HH-MM-SS"))_$(exp_name)")

println("Storing results in: $target_dir")

println("#############################")

# create plot of the observation signal
ps = plot(;title="Observation signal", xlabel="time [s]", ylabel="Water surface height [m]")
plot!(ps, obs_data.t, obs_data.H; label=reshape(["Sensor $i" for i in obs_config.sensor_id], 1,length(obs_config.sensor_id)))

###############################################################################
# Setup the forward model, likelihood, prior and proposal for MCMC sampling   #
###############################################################################

# define forward model
solver = swe_solver(sim_config)
forward_model_uncorrelated(params) = simulation(params, solver, obs_data, correlated=false)
forward_model_correlated(params) = simulation(params, solver, obs_data, correlated=true)

# Defining likelihood distribution
likelihood_σ = mcmc_config.likelihood_σ
if likelihood_σ == 0.0
    flat_signal = forward_model_uncorrelated(zeros(mcmc_config.dim))
    residual = obs_data.H .- flat_signal
    likelihood_σ = vec(std(residual, dims=1)) # set to std of flat signal residuals
    spatial_cov_mat = cov(residual)
    println("Calculated likelihood std from flat signal residuals: $(likelihood_σ)")
end

timesteps = length(obs_data.t)

acf = autocor(residual, 0:timesteps-1)
# estimate dominant frequency of the residual signal for correlation fit initial guess
Rf = fft(residual )#.- mean(residual))
Sf = abs.(Rf).^2

# positive frequencies only
idx = 1:div(timesteps,2)

max_fr_id = argmax(Sf[idx, :], dims=1)

freqs = fftfreq(timesteps, 1/sim_config.timestep)
fr = [freqs[idx[id_max[1]]] for id_max in max_fr_id]

# kernel model for time correlation
# damped oscillation model
damped_oscillation(τ, p) = exp.(-abs.(τ)./p[1]) .* cos.(2π*p[2].*τ)

p0 = [2.0, mean(fr)] # initial guess for parameters

p_mean = zeros(length(p0))

# fit the model to each column of the autocorrelation function
for r in eachcol(acf)
    p_fit = curve_fit(damped_oscillation, obs_data.t, r, p0)
    println("Fitted parameters for residual: $(p_fit.param)")
    global p_mean += p_fit.param
end

p_mean = p_mean ./ size(residual, 2)

C_temporal = damped_oscillation(obs_data.t .- obs_data.t', p_mean)
C_spatial = spatial_cov_mat

C_total = kron(C_spatial, C_temporal)

residual = vec(residual)

likelihood_dist = MvNormal(zeros(length(residual)), PDMat(C_total))

# define prior distributions
prior_dist = Vector{Distribution}()
for prior_type in lowercase.(prior_settings.type)
    if prior_type == "smooth"
        smooth_kernel = SqExpMvNormal(mcmc_config.dim, prior_settings.lengthscale, prior_settings.var)
        push!(prior_dist, MvNormal(smooth_kernel))
    elseif prior_type == "sparse"
        push!(prior_dist, Cauchy(prior_settings.loc, prior_settings.scale))
    end
end
println("Using prior distribution: $(prior_settings.type)")

# define proposal distribution
proposal_kernel = PDMat(Matrix(I, mcmc_config.dim, mcmc_config.dim))
if proposal_settings.kernel == "smooth"
    kernel = SqExpMvNormal(mcmc_config.dim, proposal_settings.lengthscale, proposal_settings.var)
    proposal_kernel = MvNormal(kernel).Σ
end

proposal = RandomWalkProposal(mcmc_config.γ, proposal_kernel)

if lowercase(proposal_settings.type) == "pcn"
    proposal_kernel = Matrix{Float64}(I, mcmc_config.dim, mcmc_config.dim)
    proposal = pCNProposal(mcmc_config.γ[1], PDMat(proposal_kernel))
end
println("Using proposal: $(proposal_settings.type)")
# add newly calculated information to config
toml_config["sampler"]["likelihood_var"] = likelihood_σ

# Deprecated: (Now directly specified in config file by user)
# toml_config["sampler"]["prior"] = "$prior_dist"
# toml_config["sampler"]["proposal"] = "$proposal_kernel"
# toml_config["sampler"]["proposal_type"] = "$(typeof(proposal))"

# Put everything into the MCMC model
pos = Posterior(prior_dist, likelihood_dist)
model = MCMCModel(pos, forward_model_correlated, obs_data, proposal)

# Define initial parameters
init_θ = mcmc_config.initial_θ
if isempty(init_θ)
    init_θ = [zeros(mcmc_config.dim) for i in 1:mcmc_config.n_chains]
    toml_config["sampler"]["init"] = init_θ
    xs = collect(range(sim_config.xbounds[1], sim_config.xbounds[2], length=mcmc_config.dim))
    inip = plot(xs, init_θ[1])
end
println("#############################")

println("Start $(mcmc_config.n_chains) chains with $(mcmc_config.n) samples: \n#############################" )

###############################################################################
# Run the MCMC sampling                                                       #
###############################################################################

chains = []
for i in 1:mcmc_config.n_chains
    chain = sample_chain(model, mcmc_config, init_θ[i], verbose=true, logging=Progress(mcmc_config.n))
    push!(chains, chain)
end

println("Chains finished \n#############################" )

###############################################################################
# Store the chains and create diagnostic plots                                #
###############################################################################

if store_exp
    mkpath(target_dir)
    mkpath(joinpath(target_dir,"plots"))
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

    cd("./plots")
    savefig(ps, "observation_signal.png")
    savefig(inip, "initial_parameters.png")
    savefig(pc, "chain.png")
    savefig(plp, "logp.png")
    savefig(pll, "loglikelihood.png")
    savefig(plprior, "logprior.png")
    savefig(pla, "acceptance_rate.png")
end