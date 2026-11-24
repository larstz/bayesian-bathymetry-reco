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

seed = get(toml_config["sampler"], "seed", 1910) # each chain takes different steps
Random.seed!(seed)
println("Random seed: $(seed)")

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
config_tag = splitext(basename(config_file))[1]
job_tag    = haskey(ENV, "SLURM_ARRAY_JOB_ID") ? "_job$(ENV["SLURM_ARRAY_JOB_ID"])-$(ENV["SLURM_ARRAY_TASK_ID"])" : ""
# Directory structure for storing results experiment/sensors/prior/proposal/stepsize/timestamp_expname
target_dir = joinpath(io_config.output_dir,
                      exp_type,
                      "sensor-"*join(obs_config.sensor_id, "-"),
                      "prior-"*join(prior_settings.type,"-"),
                      "proposal-"*proposal_settings.type * "-" * proposal_settings.kernel,
                      "stepsize-"*join(string.(mcmc_config.γ),"-"),
                      "$(Dates.format(now(), "Y-mm-dd-HH-MM-SS"))_$(exp_name)_$(config_tag)_$(job_tag)")
isdir(target_dir) && error("result directory exists: $(target_dir)")
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
# bathymetry-agnostic error model: residual w.r.t. flat reference bathymetry (no bathymetry knowledge)
flat_signal = forward_model_uncorrelated(zeros(mcmc_config.dim))
residual    = obs_data.H .- flat_signal
timesteps   = length(obs_data.t)
dt          = obs_data.t[2] - obs_data.t[1]

C_spatial   = cov(residual)                                   # variance per sensor + covariance between sensors
likelihood_σ = sqrt.(diag(C_spatial))
println("likelihood std per sensor (agnostic): $(likelihood_σ)")

# temporal correlation: damped oscillation, fitted jointly to all sensors up to ~one wave period
maxlag = min(300, timesteps ÷ 3)
lags   = collect(0:maxlag) .* dt
acf    = autocor(residual, 0:maxlag)                          # (maxlag+1) × sensors

Sf  = abs.(fft(residual, 1)) .^ 2                             # initial guess: dominant frequency
idx = 2:div(timesteps, 2)
fr  = fftfreq(timesteps, 1 / dt)[idx][[argmax(Sf[idx, i]) for i in axes(Sf, 2)]]
p0  = [1.0, mean(fr)]

damped_oscillation(τ, p) = exp.(-abs.(τ) ./ p[1]) .* cos.(2π * p[2] .* τ)
p_fit = curve_fit(damped_oscillation, repeat(lags, size(acf, 2)), vec(acf), p0).param
println("temporal kernel: ℓ = $(p_fit[1]) s, f = $(p_fit[2]) Hz")

C_temporal = damped_oscillation(obs_data.t .- obs_data.t', p_fit)
C_total    = Symmetric(kron(C_spatial, C_temporal))
likelihood_dist = MvNormal(zeros(length(residual)), PDMat(Matrix(C_total)))

# record what was used
toml_config["likelihood"] = Dict("type" => "agnostic (residual w.r.t. flat bathymetry)",
    "sigma" => likelihood_σ, "spatial_cov" => [collect(r) for r in eachrow(C_spatial)],
    "kernel" => "exp(-|τ|/ℓ) cos(2π f τ)", "ell" => p_fit[1], "f" => p_fit[2], "maxlag_s" => maxlag * dt)


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
fixed = [1, mcmc_config.dim]  # mask fixed parameters (first and last)
proposal = MaskedProposal(proposal, fixed)  # mask fixed parameters
if lowercase(proposal_settings.type) == "pcn"
    proposal_kernel = Matrix{Float64}(I, mcmc_config.dim, mcmc_config.dim)
    proposal = pCNProposal(mcmc_config.γ[1], PDMat(proposal_kernel))
end
println("Using proposal: $(proposal_settings.type)")

# Deprecated: (Now directly specified in config file by user)
# toml_config["sampler"]["prior"] = "$prior_dist"
# toml_config["sampler"]["proposal"] = "$proposal_kernel"
# toml_config["sampler"]["proposal_type"] = "$(typeof(proposal))"

# Put everything into the MCMC model
pos = Posterior(prior_dist, likelihood_dist)
model = MCMCModel(pos, forward_model_correlated, obs_data, proposal)

# Define initial parameters
init_θ = mcmc_config.initial_θ
xs = collect(range(sim_config.xbounds[1], sim_config.xbounds[2], length=mcmc_config.dim))
if isempty(init_θ)
    #init_θ = [zeros(mcmc_config.dim) for i in 1:mcmc_config.n_chains]
    pr       = toml_config["sampler"]["prior"]
    G        = MvNormal(SqExpMvNormal(mcmc_config.dim, Int(pr["lengthscale"]), pr["var"]))   # smooth (Gaussian) prior part
    init_rng = MersenneTwister(get(toml_config["sampler"], "init_seed", seed))
    init_θ   = [rand(init_rng, G) for _ in 1:mcmc_config.n_chains]
end
all(length.(init_θ) .== mcmc_config.dim) || error("initial states must have length $(mcmc_config.dim)")

# fix boundary parameters to zero
foreach(θ -> θ[fixed] .= 0, init_θ)
toml_config["sampler"]["init"] = init_θ

toml_config["sampler"]["mask_fixed"] = fixed
toml_config["run_info"] = Dict("git_commit" => readchomp(`git rev-parse --short HEAD`),
                               "uncommitted_changes" => !isempty(readchomp(`git status --porcelain`)),
                               "julia_version" => string(VERSION), "date" => string(now()))

inip = plot(xs, init_θ, label=permutedims(["chain $(i)" for i in eachindex(init_θ)]), xlabel="x [m]", ylabel="b [m]")
println("#############################")

println("Start $(mcmc_config.n_chains) chains with $(mcmc_config.n) samples: \n#############################" )

###############################################################################
# Run the MCMC sampling                                                       #
###############################################################################
ckpt_every = get(toml_config["output"], "checkpoint_every", 5000)
ckpt_dir   = joinpath(toml_config["output"]["path"], "checkpoints")
mkpath(ckpt_dir)
run_tag    = splitext(basename(config_file))[1]            # unique per array job (one config each)

chains = []
for i in 1:mcmc_config.n_chains
    ckpt_file = joinpath(ckpt_dir, "$(run_tag)_chain_$(i).jls")
    function write_ckpt(iter, ch)
        tmp = ckpt_file * ".tmp"
        serialize(tmp, Matrix(ch))                         # write to a temporary file first,
        mv(tmp, ckpt_file; force=true)                     # then replace: never a half-written checkpoint
    end
    chain = sample_chain(model, mcmc_config, init_θ[i], verbose=true, logging=Progress(mcmc_config.n),
                         callback_every=ckpt_every, callback=write_ckpt)
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