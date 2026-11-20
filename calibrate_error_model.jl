# Calibrate the error model on the flat-bed runs, store it, and check it on the inverted run.
# Usage (repository root):  julia calibrate_error_model.jl <single-run config.toml>
using Pkg
Pkg.activate(".")

using BathymetryReco
using TOML, Serialization, Statistics, LinearAlgebra, PDMats

config = load_config(TOML.parsefile(abspath(ARGS[1])))
sim_config, mcmc_config, obs_config = config.sim_params, config.mcmc_params, config.obs_settings
calib_files = ["data/experiment_data/without_bathymetry/Heat$(k).txt" for k in 1:20]
out_file    = "data/results/likelihood/error_model.jls"
mkpath(dirname(out_file))

sim_config.bc_file == obs_config.path || @warn "bc_file ≠ observation path"
obs_config.path in calib_files && error("the inverted run must not be a calibration run")

# calibration (flat runs only)
em = calibrate_error_model(sim_config, obs_config.sensor_id, calib_files; dim=mcmc_config.dim)
serialize(out_file, em)
println("α = $(em.α), ℓ = $(em.ℓ) s, f = $(em.f) Hz, R = $(round.(em.R, digits=3))")

# covariance of the inverted run and effective number of samples for a wave-shaped signal
obs = load_observation(obs_config.path, sim_config.tstart, sim_config.tinterval, sensor_id=obs_config.sensor_id)
Σ = error_covariance(em, sim_config, obs; dim=mcmc_config.dim)
v = vec(simulation(zeros(mcmc_config.dim), sim_config, obs, correlated=false) .- obs.H[1:1, :])
println("n_eff = $(round(length(v) * dot(v, Σ \ v) / dot(v, v ./ diag(Σ)), digits=1)) of $(length(v))")

# DIAGNOSTIC ONLY: whitened residual at the measured bathymetry (wave part: rms ≈ 1 expected)
xs = collect(range(sim_config.xbounds[1], sim_config.xbounds[2], length=mcmc_config.dim))
r  = obs.H .- simulation(exp_bathymetry(xs), sim_config, obs, correlated=false)
S, σ0 = error_std(em, sim_config, obs; dim=mcmc_config.dim)
w  = Σ.chol.L \ vec(r)
nt = size(r, 1)
for i in axes(r, 2)
    wi, wave = w[(i-1)*nt+1:i*nt], S[:, i] .> 3σ0[i]
    lag1(x) = cor(x[1:end-1], x[2:end])
    println("sensor $(obs_config.sensor_id[i]): wave part rms = $(round(sqrt(mean(abs2, wi[wave])), digits=2)), ",
            "lag-1 = $(round(lag1(wi[wave]), digits=2)) | quiet part rms = $(round(sqrt(mean(abs2, wi[.!wave])), digits=2))")
end