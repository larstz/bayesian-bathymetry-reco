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

using StatsBase
using LsqFit
using FFTW
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
obs_config = config.obs_settings




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


###############################################################################
# Setup the forward model, likelihood, prior and proposal for MCMC sampling   #
###############################################################################

# define forward model
solver = swe_solver(sim_config)
forward_model(params, correlated::Bool=true) = simulation(params, solver, obs_data, correlated=correlated)

# Defining likelihood distribution
likelihood_σ = mcmc_config.likelihood_σ
if likelihood_σ == 0.0
    flat_signal = forward_model(zeros(mcmc_config.dim), false)
    residual = obs_data.H .- flat_signal
    likelihood_σ = vec(std(residual, dims=1)) # set to std of flat signal residuals
    spatial_cov_mat = cov(residual)
    println("Calculated likelihood std from flat signal residuals: $(likelihood_σ)")
end

timesteps = length(obs_data.t)

acf = autocor(residual, 0:timesteps-1)
# estimate dominant frequency of the residual signal for correlation fit initial guess
Rf = fft(residual.- mean(residual))
Sf = abs.(Rf).^2

# positive frequencies only
idx = 1:div(timesteps,2)

max_fr_id = argmax(Sf[idx, :], dims=1)

freqs = fftfreq(timesteps, 1/sim_config.timestep)

freq = (0:timesteps-1) ./ (timesteps*sim_config.timestep)
fr = [freqs[idx[id_max[1]]] for id_max in max_fr_id]

# kernel model for time correlation
# damped oscillation model
model(τ, p) = exp.(-abs.(τ)./p[1]) .* cos.(2π*p[2].*τ)

p0 = [2.0, mean(fr)] # initial guess for parameters

p_mean = zeros(length(p0))

# fit the model to each column of the autocorrelation function
for r in eachcol(acf)
    p_fit = curve_fit(model, obs_data.t, r, p0)
    println("Fitted parameters for residual: $(p_fit.param)")
    p_mean += p_fit.param
end

p_mean = p_mean ./ size(residual, 2)

C_temporal = model(obs_data.t .- obs_data.t', p_mean)
C_spatial = spatial_cov_mat

C_total = kron(C_spatial, C_temporal)

residual = vec(residual)

likelihood_dist = MvNormal(zeros(length(residual)), PDMat(C_total))

plot(rand(likelihood_dist, 10), label="Random samples from correlated likelihood")