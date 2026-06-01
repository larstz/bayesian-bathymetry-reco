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

using Random

Random.seed!(1910)

ENV["GKSwstype"]="nul"

###############################################################################
# Load the configuration                                                      #
###############################################################################

println("#############################\nRead in config file" )
if isempty(ARGS)
    config_file = abspath("./configs/configtest.toml")
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
forward_model(params) = simulation(params, solver, obs_data)

# Defining likelihood distribution
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
for prior_type in lowercase.(prior_settings.type)
    if prior_type == "smooth"
        smooth_kernel = SqExpMvNormal(mcmc_config.dim, prior_settings.lengthscale, prior_settings.var)
        push!(prior_dist, MvNormal(smooth_kernel))
    elseif prior_type == "sparse"
        push!(prior_dist, Cauchy(prior_settings.loc, prior_settings.scale))
    end
end
println("Using prior distribution: $(prior_settings.type)")

# add newly calculated information to config
toml_config["sampler"]["likelihood_var"] = likelihood_σ

# Put everything into the MCMC model
pos = Posterior(prior_dist, likelihood_dist)
# proposal only needed to fit with the MCMCModel type but not needed in TMCMC
proposal = Normal(0.0,0.0)
model = MCMCModel(pos, forward_model, obs_data, proposal)

# Define initial parameters
init_θ = mcmc_config.initial_θ
if isempty(init_θ)
    #init_θ = zeros(mcmc_config.n,mcmc_config.dim) # using only zero vectors does not make sense for tmcmc
    if length(prior_dist) == 1
        @info("Only single prior distribution was supplied - using it to sample initial guess for all $(mcmc_config.dim) parameters!")
        init_θ = rand(prior_dist[1],(mcmc_config.n,mcmc_config.dim))
    elseif length(prior_dist) == mcmc_config.dim
        @info("Individual prior distribution for each of the $(mcmc_config.dim) parameters was supplied!")
        init_θ = reduce(hcat,rand.(prior_dist2,(mcmc_config.n)))
    else
        @error("Prior_dist vector needs to be either length 1 or $(mcmc_config.dim)!")
    end
    toml_config["sampler"]["init"] = init_θ
    xs = collect(range(sim_config.xbounds[1], sim_config.xbounds[2], length=mcmc_config.dim))
    mean_init_θ = vec(mean(init_θ ,dims=1))
    inip = plot(xs, mean_init_θ, label="Mean initialization")
end
println("#############################")


###############################################################################
# Run the MCMC sampling                                                       #
###############################################################################
println("Start TMCMC with $(mcmc_config.n) samples: \n#############################" )

parameters, S = transitional_mcmc(model, mcmc_config, init_θ, verbose=true, logging=Progress(mcmc_config.n))

println("TMCMC finished \n#############################" )

###############################################################################
# Store the chains and create diagnostic plots                                #
###############################################################################

if store_exp
    @warn("Currently missing!")
end