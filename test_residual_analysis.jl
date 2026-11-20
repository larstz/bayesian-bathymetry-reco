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
using Random
using CSV
using DataFrames

Random.seed!(1910)

ENV["GKSwstype"]="nul"

###############################################################################
# Error model for the likelihood                                              #
#                                                                             #
#   residual = H_obs - F(b) ~ N(0, Σ),                                        #
#   Σ = S (R ⊗ C_t) S + diag(σ0²)                                             #
#                                                                             #
#   S   = diag(α_i s_i(t)): discrepancy std, proportional to the local wave   #
#         amplitude s_i(t) (running rms over one wave period) of the flat-bed #
#         prediction for the inverted run (own boundary record, independent   #
#         of b, so Σ is fixed during sampling)                                #
#   R   : correlation between sensors at zero lag                             #
#   C_t : temporal correlation, damped cosine exp(-|τ|/ℓ) cos(2π f τ)         #
#   σ0  : measurement noise floor of the inverted run (still water, t<tstart) #
#                                                                             #
# α, R, ℓ, f are estimated from the flat-bed calibration runs only            #
# (known bathymetry b = 0), never from the run that is inverted.              #
###############################################################################

###############################################################################
# Load the configuration                                                      #
###############################################################################

cov_model = :separable                                    # :chain (error propagates downstream) or :separable
maxlag_xc = 250                                        # cross-correlation search range ±2.5 s

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

# calibration runs (flat bed)
calib_dir  = "data/experiment_data/without_bathymetry"
calib_runs = 1:20
calib_file(k) = joinpath(calib_dir, "Heat$(k).txt")

# envelope and wave-part settings
T_wave  = 1 / 0.35                                    # flap period [s]
w_env   = 2 * round(Int, T_wave / sim_config.timestep / 2) + 1   # one period, odd
s_quiet = 1e-4                                        # envelope < 0.1 mm: no wave yet

out_dir = "data/results/likelihood"
mkpath(out_dir)

println("##############################\nLoad experiment data")

###############################################################################
# Load the observation data (the run that is inverted)                        #
###############################################################################

obs_config.real_data || error("This error model is estimated from experimental calibration runs; use real data.")
occursin("mean", obs_config.path) &&
    @warn "Observation path points to an averaged file; the error model assumes a single run: $(obs_config.path)"
sim_config.bc_file == obs_config.path ||
    @warn "bc_file and observation path differ; they should be the same run:\n  bc_file = $(sim_config.bc_file)\n  obs     = $(obs_config.path)"

obs_data = load_observation(obs_config.path, sim_config.tstart, sim_config.tinterval,
                            sensor_id=obs_config.sensor_id, noise_var=obs_config.noise_var)
timesteps = length(obs_data.t)
n_sensors = length(obs_config.sensor_id)

###############################################################################
# Helpers                                                                     #
###############################################################################

# same setup with another boundary record (fields in declaration order of SimulationSetup)
with_bc(s::SimulationSetup, file) = SimulationSetup(s.xbounds, s.sensor_pos, s.timestep, s.nx, s.tstart,
                                                    s.tinterval, s.g, s.kappa, s.dealias, s.scenario,
                                                    file, "flat")

# centred moving average with window w (odd); window shrinks at the edges
function movmean(x::AbstractVector, w::Int)
    n, h = length(x), w ÷ 2
    c = cumsum([0.0; x])
    return [(c[min(n, i + h) + 1] - c[max(1, i - h)]) / (min(n, i + h) - max(1, i - h) + 1) for i in 1:n]
end
running_rms(x, w) = sqrt.(movmean(x .^ 2, w))

# local wave amplitude of a model prediction (time × sensor); still water taken from the first sample
envelope(H) = hcat([running_rms(H[:, i] .- H[1, i], w_env) for i in axes(H, 2)]...)

# propagation ("chain") correlation between sensors: block (i,j), j ≥ i, = A[i,j] * ρ(t_j - t_i - D[i,j])
function chain_correlation(t, p, a, Δ)
    ns = length(a) + 1
    A = ones(ns, ns); D = zeros(ns, ns)
    for i in 1:ns, j in i+1:ns
        A[i, j] = prod(a[i:j-1]); D[i, j] = sum(Δ[i:j-1])
    end
    nt = length(t)
    C = zeros(ns * nt, ns * nt)
    for i in 1:ns, j in i:ns
        B = A[i, j] .* model(t' .- t .- D[i, j], p)          # rows: times at sensor i, columns: sensor j
        C[(i-1)*nt+1:i*nt, (j-1)*nt+1:j*nt] = B
        j > i && (C[(j-1)*nt+1:j*nt, (i-1)*nt+1:i*nt] = B')
    end
    return Symmetric(C)
end

flat_params = zeros(mcmc_config.dim)

###############################################################################
# Calibration: flat-bed residuals (each run with its own boundary record)     #
###############################################################################

println("##############################\nFlat-bed calibration runs")
r = Dict{Int, Matrix{Float64}}()
s = Dict{Int, Matrix{Float64}}()
@showprogress for k in calib_runs
    obs_k = load_observation(calib_file(k), sim_config.tstart, sim_config.tinterval,
                             sensor_id=obs_config.sensor_id, noise_var=obs_config.noise_var)
    H_k = simulation(flat_params, with_bc(sim_config, calib_file(k)), obs_k, correlated=false)
    r[k] = obs_k.H .- H_k
    s[k] = envelope(H_k)
end

###############################################################################
# Size of the discrepancy: σ_i(t)² = σ0² + α_i² s_i(t)²                        #
###############################################################################

σ0_flat = zeros(n_sensors)
α = zeros(n_sensors)
α_grid = 0.0:0.002:1.5
for i in 1:n_sensors
    rq = vcat([r[k][s[k][:, i] .< s_quiet, i] for k in calib_runs]...)
    σ0_flat[i] = sqrt(mean(abs2, rq))                       # floor of the (less calm) flat runs
    rr = vcat([r[k][:, i] for k in calib_runs]...)
    ss = vcat([s[k][:, i] for k in calib_runs]...)
    nll(a) = sum(log.(σ0_flat[i]^2 .+ a^2 .* ss .^ 2) ./ 2 .+ rr .^ 2 ./ (2 .* (σ0_flat[i]^2 .+ a^2 .* ss .^ 2)))
    α[i] = α_grid[argmin(nll.(α_grid))]
    (α[i] == first(α_grid) || α[i] == last(α_grid)) && @warn "sensor $(i): α at grid limit"
end
println("Discrepancy size: α = $(α), flat-run floor σ0 = $(σ0_flat .* 1e3) mm")

###############################################################################
# Standardized residuals and their correlation (wave part only)              #
###############################################################################

σ_flat(k) = sqrt.(σ0_flat' .^ 2 .+ α' .^ 2 .* s[k] .^ 2)
z = Dict(k => r[k] ./ σ_flat(k) for k in calib_runs)
first_wave = Dict(k => [findfirst(s[k][:, i] .> s_quiet) for i in 1:n_sensors] for k in calib_runs)

# temporal autocorrelation per sensor, averaged over runs (no demeaning: the mean is part of the error)
acf_raw(x, maxlag) = [sum(x[1:end-k] .* x[1+k:end]) / sum(abs2, x) for k in 0:maxlag]
seg_len = minimum(timesteps - first_wave[k][i] + 1 for k in calib_runs, i in 1:n_sensors)
maxlag  = min(300, seg_len ÷ 2)
lags    = (0:maxlag) .* sim_config.timestep
acf     = hcat([mean(acf_raw(z[k][first_wave[k][i]:end, i], maxlag) for k in calib_runs) for i in 1:n_sensors]...)

# kernel model for time correlation: damped oscillation, one common kernel for all sensors (separable model)
#model(τ, p) = exp.(-abs.(τ) ./ p[1]) .* cos.(2π * p[2] .* τ)
#model(τ, p) = (1 .+ sqrt(3) .* abs.(τ) ./ p[1]) .* exp.(-sqrt(3) .* abs.(τ) ./ p[1]) .* cos.(2π * p[2] .* τ) # Matern 3/2 kernel with oscillation
model(τ, p) = (1 .+ sqrt(5) .* abs.(τ) ./ p[1] .+ 5 .* τ .^ 2 ./ (3 .* p[1] .^ 2)) .*
              exp.(-sqrt(5) .* abs.(τ) ./ p[1]) .* cos.(2π * p[2] .* τ) # Matern 5/2 kernel with oscillation
p0 = [1.0, 1 / T_wave]                                      # initial guess: ℓ = 1 s, wave frequency
for i in 1:n_sensors
    println("  sensor $(obs_config.sensor_id[i]): fitted [ℓ, f] = $(curve_fit(model, lags, acf[:, i], p0).param)")
end
p_fit = curve_fit(model, repeat(lags, n_sensors), vec(acf), p0).param      # joint fit -> common kernel
println("Common temporal kernel: ℓ = $(p_fit[1]) s, f = $(p_fit[2]) Hz (P = $(1 / p_fit[2]) s)")

# correlation between sensors at zero lag (common wave part, pooled over runs)
R = Matrix{Float64}(I, n_sensors, n_sensors)
for i in 1:n_sensors, j in i+1:n_sensors
    num = 0.0; ni = 0.0; nj = 0.0
    for k in calib_runs
        a = max(first_wave[k][i], first_wave[k][j])
        num += sum(z[k][a:end, i] .* z[k][a:end, j])
        ni  += sum(abs2, z[k][a:end, i]); nj += sum(abs2, z[k][a:end, j])
    end
    R[i, j] = R[j, i] = num / sqrt(ni * nj)
end
println("Correlation between sensors at zero lag:\n", round.(R, digits=3))
isposdef(R) || error("R is not positive definite")

# cross-correlation between neighbouring sensors (standardized, common wave part, mean over runs);
# c(k) = Σ z_i(t) z_j(t+k): a peak at k > 0 means the error at the downstream sensor j comes later
function xcorr_pair(i, j)
    c = zeros(2maxlag_xc + 1)
    for k in calib_runs
        s0 = max(first_wave[k][i], first_wave[k][j])
        zi, zj = z[k][s0:end, i], z[k][s0:end, j]
        n = length(zi); nrm = sqrt(sum(abs2, zi) * sum(abs2, zj))
        for (m, lag) in enumerate(-maxlag_xc:maxlag_xc)
            c[m] += (lag >= 0 ? sum(zi[1:n-lag] .* zj[1+lag:n]) : sum(zi[1-lag:n] .* zj[1:n+lag])) / nrm
        end
    end
    return c ./ length(calib_runs)
end

a_chain = zeros(n_sensors - 1); Δ_chain = zeros(n_sensors - 1)
for i in 1:n_sensors-1
    c = xcorr_pair(i, i + 1)
    m = argmax(c)
    a_chain[i] = c[m]
    Δ_chain[i] = (m - maxlag_xc - 1) * sim_config.timestep
end
println("Propagation between sensors: a = $(round.(a_chain, digits=3)), Δt = $(Δ_chain) s")
all(0 .< a_chain .< 1) || @warn "peak cross-correlation outside (0, 1): $(a_chain)"

###############################################################################
# Covariance for the inverted run                                             #
###############################################################################

# envelope from the flat-bed prediction with the inverted run's own boundary record (independent of b)
s_inf  = envelope(simulation(flat_params, sim_config, obs_data, correlated=false))
σ0_inf = vec(obs_data.noise_std)                          # still-water noise of the inverted run
length(σ0_inf) == n_sensors || error("noise_std has unexpected length $(length(σ0_inf))")

C_temporal = model(obs_data.t .- obs_data.t', p_fit)
Sv = vec(s_inf .* α')                                     # same stacking as vec(residual): sensor by sensor
#C_total = Symmetric((Sv * Sv') .* kron(R, C_temporal) + Diagonal(repeat(σ0_inf .^ 2, inner=timesteps)))
C_corr = cov_model == :chain ? chain_correlation(obs_data.t, p_fit, a_chain, Δ_chain) :
                               Symmetric(kron(R, C_temporal))
C_total = Symmetric((Sv * Sv') .* C_corr + Diagonal(repeat(σ0_inf .^ 2, inner=timesteps)))

likelihood_dist = MvNormal(zeros(n_sensors * timesteps), PDMat(Matrix(C_total)))

# information lost to correlation for a wave-shaped signal (the flat-bed prediction):
# effective number of samples  n * (vᵀ Σ⁻¹ v) / (vᵀ diag(Σ)⁻¹ v)
v = vec(simulation(flat_params, sim_config, obs_data, correlated=false) .- obs_data.H[1:1, :])
n_eff = length(v) * dot(v, C_total \ v) / dot(v, v ./ diag(C_total))
println("Effective number of samples for a wave-shaped signal: $(round(n_eff, digits=1)) of $(length(v))")

###############################################################################
# Store and plot                                                              #
###############################################################################

err_model = (; cov_model, a_chain, Δ_chain, α, σ0_flat, σ0_inf, R, ℓ=p_fit[1], f=p_fit[2], w_env, s_quiet, n_eff,
             calib_runs=collect(calib_runs), obs_path=obs_config.path, config_file, date=now())
tag = splitext(basename(obs_config.path))[1]
serialize(joinpath(out_dir, "error_model_$(tag)_$(cov_model).jls"), err_model)
CSV.write(joinpath(out_dir, "error_model_$(tag)_$(cov_model).csv"),
          DataFrame(sensor=obs_config.sensor_id, alpha=α, sigma0_flat_mm=σ0_flat .* 1e3,
                    sigma0_inf_mm=σ0_inf .* 1e3, ell_s=p_fit[1], f_Hz=p_fit[2]))

p1 = plot(lags, acf, label=reshape(["sensor $(i)" for i in obs_config.sensor_id], 1, :), lw=2,
          xlabel="lag [s]", title="autocorrelation (standardized, wave part)")
plot!(p1, lags, model(lags, p_fit), ls=:dash, color=:black, label="common kernel")
p2 = plot(obs_data.t, reshape(sqrt.(diag(C_total)), timesteps, n_sensors) .* 1e3,
          label=reshape(["sensor $(i)" for i in obs_config.sensor_id], 1, :), xlabel="t [s]", ylabel="σ(t) [mm]",
          title="error std of the inverted run")
p3 = plot(rand(likelihood_dist, 3) .* 1e3, label=false, ylabel="[mm]", title="samples from the error model")
savefig(plot(p1, p2, p3, layout=(3, 1), size=(900, 1000)), joinpath(out_dir, "error_model_$(tag)_$(cov_model).pdf"))
println("Error model written to $(out_dir)")

# DIAGNOSTIC ONLY (uses the measured bathymetry): does the error model fit the inverted run?
equi_x = collect(range(sim_config.xbounds[1], sim_config.xbounds[2], length=mcmc_config.dim))
b_true_params = exp_bathymetry(equi_x)                    # true bump in the inference parametrization
r_b = obs_data.H .- simulation(b_true_params, sim_config, obs_data, correlated=false)

σ_inf = reshape(sqrt.(diag(C_total)), timesteps, n_sensors)
println("rms of r/σ(t) per sensor: ", round.([sqrt(mean(abs2, r_b[:, i] ./ σ_inf[:, i])) for i in 1:n_sensors], digits=2))

w = cholesky(Matrix(C_total)).L \ vec(r_b)                # whitened residual
println("whitened residual: rms = ", round(sqrt(mean(abs2, w)), digits=2),
        ", lag-1 autocorrelation = ", round(autocor(w, [1])[1], digits=2))

# whitened residual, split into quiet part and wave part per sensor
lag1(x) = cor(x[1:end-1], x[2:end])
for i in 1:n_sensors
    blk  = (i-1)*timesteps+1:i*timesteps
    wave = Sv[blk] .> 3 .* σ0_inf[i]                       # discrepancy dominates the floor
    wq, ww = w[blk][.!wave], w[blk][wave]
    println("sensor $(obs_config.sensor_id[i]): quiet  rms = $(round(sqrt(mean(abs2, wq)), digits=2)), lag-1 = $(round(lag1(wq), digits=2))",
            " | wave  rms = $(round(sqrt(mean(abs2, ww)), digits=2)), lag-1 = $(round(lag1(ww), digits=2))")
end