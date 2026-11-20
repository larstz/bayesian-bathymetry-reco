###############################################################################
# Error model: residual = H_obs - F(b) ~ N(0, Σ),                             #
#   Σ = S (R ⊗ C_t) S + diag(σ0²)                                             #
#   S   = diag(α_i s_i(t)), s_i(t): running rms (one wave period) of the      #
#         flat-bed prediction of the inverted run (independent of b)          #
#   C_t : Matérn-5/2 × cos(2π f τ),  R: zero-lag correlation between sensors  #
#   σ0  : still-water noise of the inverted run                               #
# α, ℓ, f, R are estimated from flat-bed calibration runs only (b = 0).       #
###############################################################################

export calibrate_error_model, error_std, error_covariance

matern52_cos(τ, ℓ, f) = (r = sqrt(5) * abs(τ) / ℓ; (1 + r + r^2 / 3) * exp(-r) * cos(2π * f * τ))

function movmean(x, w)                       # centred moving average, window shrinks at the edges
    n, h = length(x), w ÷ 2
    c = cumsum([0.0; x])
    return [(c[min(n, i + h) + 1] - c[max(1, i - h)]) / (min(n, i + h) - max(1, i - h) + 1) for i in 1:n]
end
envelope(H, w) = hcat([sqrt.(movmean((H[:, i] .- H[1, i]) .^ 2, w)) for i in axes(H, 2)]...)

with_bc(s::SimulationSetup, file) = SimulationSetup(s.xbounds, s.sensor_pos, s.timestep, s.nx, s.tstart, s.tinterval,
                                                    s.g, s.kappa, s.dealias, s.scenario, file, s.bathy_name)

"""
    calibrate_error_model(sim_config, sensor_id, calib_files; dim, T_wave=1/0.35, s_quiet=1e-4)

Estimate α, ℓ, f and R from flat-bed runs (each simulated with b = 0 and its own boundary record).
Returns a NamedTuple; store it with `serialize`.
"""
function calibrate_error_model(sim_config::SimulationSetup, sensor_id, calib_files; dim, T_wave=1/0.35, s_quiet=1e-4)
    r, s, w = Matrix{Float64}[], Matrix{Float64}[], 0
    @showprogress for file in calib_files
        obs = load_observation(file, sim_config.tstart, sim_config.tinterval; sensor_id=sensor_id)
        w = 2 * round(Int, T_wave / (obs.t[2] - obs.t[1]) / 2) + 1
        H = simulation(zeros(dim), with_bc(sim_config, file), obs; correlated=false)
        push!(r, obs.H .- H); push!(s, envelope(H, w))
    end
    nt, ns = size(r[1]); dt = sim_config.timestep

    # size: floor from the quiet part, α by maximum likelihood (floor fixed)
    σ0, α = zeros(ns), zeros(ns)
    for i in 1:ns
        σ0[i] = sqrt(mean(abs2, vcat([rk[sk[:, i] .< s_quiet, i] for (rk, sk) in zip(r, s)]...)))
        rr, ss = vcat([rk[:, i] for rk in r]...), vcat([sk[:, i] for sk in s]...)
        nll(a) = sum(log.(σ0[i]^2 .+ a^2 .* ss .^ 2) ./ 2 .+ rr .^ 2 ./ (2 .* (σ0[i]^2 .+ a^2 .* ss .^ 2)))
        α[i] = argmin(nll, 0.0:0.002:1.5)
    end

    # standardized residuals over the wave part
    z  = [rk ./ sqrt.(σ0' .^ 2 .+ α' .^ 2 .* sk .^ 2) for (rk, sk) in zip(r, s)]
    fw = [[findfirst(sk[:, i] .> s_quiet) for i in 1:ns] for sk in s]    # first wave sample per run, sensor

    # temporal kernel: least-squares fit to the mean autocorrelation (grid search, all sensors jointly)
    maxlag = min(300, minimum(nt - q[i] + 1 for q in fw, i in 1:ns) ÷ 2)
    acf(x) = [dot(x[1:end-k], x[1+k:end]) / dot(x, x) for k in 0:maxlag]
    ρ = vcat([mean(acf(zk[q[i]:end, i]) for (zk, q) in zip(z, fw)) for i in 1:ns]...)
    τ = repeat((0:maxlag) .* dt, ns)
    sse((ℓ, f)) = sum(abs2, ρ .- matern52_cos.(τ, ℓ, f))
    ℓ, f = argmin(sse, Iterators.product(0.05:0.01:4.0, 0.05:0.005:1.5))

    # correlation between sensors at zero lag (common wave part, pooled over runs)
    R = Matrix{Float64}(I, ns, ns)
    for i in 1:ns, j in i+1:ns
        seg(zk, q, m) = zk[max(q[i], q[j]):end, m]
        R[i, j] = R[j, i] = sum(dot(seg(zk, q, i), seg(zk, q, j)) for (zk, q) in zip(z, fw)) /
                            sqrt(sum(sum(abs2, seg(zk, q, i)) for (zk, q) in zip(z, fw)) *
                                 sum(sum(abs2, seg(zk, q, j)) for (zk, q) in zip(z, fw)))
    end
    isposdef(R) || error("R is not positive definite")

    return (; α, σ0_flat=σ0, ℓ, f, R, w_env=w, s_quiet, sensor_id=collect(sensor_id), calib_files=collect(calib_files))
end

"""
    error_std(em, sim_config, obs; dim)

Discrepancy std S (time × sensor) and noise floor σ0 for the run in `sim_config.bc_file` / `obs`.
"""
function error_std(em, sim_config::SimulationSetup, obs; dim)
    S = envelope(simulation(zeros(dim), sim_config, obs; correlated=false), em.w_env) .* em.α'
    return S, vec(obs.noise_std)
end

"""
    error_covariance(em, sim_config, obs; dim)

Σ = S (R ⊗ C_t) S + diag(σ0²) for `vec` of the residual (time × sensor, stacked sensor by sensor)
of the run in `sim_config.bc_file` / `obs`.
"""
function error_covariance(em, sim_config::SimulationSetup, obs; dim)
    S, σ0 = error_std(em, sim_config, obs; dim=dim)
    t, Sv = vec(obs.t), vec(S)
    Σ = (Sv * Sv') .* kron(em.R, matern52_cos.(t .- t', em.ℓ, em.f))
    Σ[diagind(Σ)] .+= repeat(σ0 .^ 2, inner=length(t))
    return PDMat(Matrix(Symmetric(Σ)))
end