using Pkg
Pkg.activate(".")
using Serialization
using Plots
using StatsPlots
using Statistics
using BathymetryReco
using MCMCChains
using Distributions
using LaTeXStrings
using CSV
using DataFrames
using DataInterpolations

include("my_theme.jl")
theme(:custom)

println("#############################\nRead in chain" )

exp = ARGS[1]

out_dir = length(ARGS) > 1 ? ARGS[2] : ""

ani = false
include_adjoint = false
plot_sensor_simulation = true
appendix = ""
chain = deserialize(joinpath(exp, "chain_1.jls"))

config = load_config(joinpath(exp, "experiment_config.toml"))
sim_config = config.sim_params
mcmc_config = config.mcmc_params
obs_config = config.obs_settings

proposal = isempty(mcmc_config.proposal.kernel) ? "normal" : mcmc_config.proposal.kernel
proposal_algo = mcmc_config.proposal.type
prior = join(mcmc_config.prior.type, "_")
stepsize = mcmc_config.γ[1]
# Load the data
if obs_config.real_data
    obs_data = load_observation(obs_config.path, sim_config.tstart, sim_config.tinterval)
    adjoint_file = "data/results/experiment_adjoint.csv"
    adjoint_label = "Adjoint Solution, NRMSE = 16.42"
else
    obs_data, exact_b = load_toy_observation(obs_config.path, obs_config.noise_var, sensor_rate=obs_config.sensor_rate)
    adjoint_file = "data/results/simulated_noise_adjoint.csv"
    adjoint_label = "Adjoint Solution, NRMSE = 10.76"
end

solver = swe_solver(sim_config)
forward(params) = simulation(params, solver, obs_data)

burnin = 1000
sample_number= 50001

bathy_before_burnin = chain[1:sample_number,1:mcmc_config.dim]
lp_before_burnin = chain[1:sample_number,mcmc_config.dim+1]
ar_before_burnin = chain[1:sample_number,end]
ll_before_burnin = chain[1:sample_number,mcmc_config.dim+2]
lprior_before_burnin = chain[1:sample_number,mcmc_config.dim+3]


bathy = chain[burnin+1:sample_number,1:mcmc_config.dim]
lp = chain[burnin+1:sample_number,mcmc_config.dim+1]
ar = chain[burnin+1:sample_number,end]

println("Acceptance rate after burn-in: ", round(mean(ar), digits=4))
println("Acceptance rate at the end of the chain: ", round(ar[end], digits=4))


xs = range(1.5,15.0,mcmc_config.dim)
if obs_config.real_data || sim_config.bathy_name == "exact_bathy"
    exact_b = exp_bathymetry(xs)
else
    exact_b = PCHIPInterpolation(exact_b, obs_data.sim_x; extrapolation = ExtrapolationType.Constant)(xs)
end

if ani
println("#############################\nCreate Gif" )

anim = @animate for (i, b) in enumerate(eachrow(bathy[1:10:end,:]))
    plot(xs, b, label="chain $i", ylims=(-0.01,0.21), xlabel="x", ylabel="b(x)",title="Bathymetry Sample progression")
    plot!(xs, exact_b, label="True Bathymetry", color=:black)
end every 10

gif(anim, exp*"/plots/chain_progression.gif", fps=10)
end
println("#############################\nCreate error plot" )

mean_bathy = vec(mean(bathy, dims=1))
median_bathy = vec(median(bathy, dims=1))
mode_bathy = vec(bathy[argmax(lp), :])
bathy_nrmse = sqrt.(mean((mean_bathy .- exact_b).^2)) ./ (maximum(exact_b) .- minimum(exact_b))*100
bathy_l2 = sqrt(sum((mean_bathy .- exact_b).^2))/sqrt(sum(exact_b.^2))*100
bathy_linf = maximum(abs.(mean_bathy .- exact_b))/maximum(exact_b)*100
mcmc_chain = Chains(bathy)
grid_error = mcse(mcmc_chain)[:, :mcse]
grid_ci_low = hpd(mcmc_chain; alpha=0.32)[:, :lower]
grid_ci_high = hpd(mcmc_chain; alpha=0.32)[:, :upper]

# effective sample size per parameter (used for t-multiplier)
n_eff = ess(mcmc_chain)[:, :ess]
# safe function for t multiplier from n_eff
function t_from_neff(ne)
    nei = max(2, floor(Int, ne))
    return quantile(TDist(nei-1), 0.975)
end
t_vals = [t_from_neff(ne) for ne in n_eff]

# mean CI (mean ± t * MCSE)
mean_ci_low = mean_bathy .- t_vals .* grid_error
mean_ci_high = mean_bathy .+ t_vals .* grid_error

result_df = DataFrame(x=xs, mean_bathy=mean_bathy, mean_ci_low=mean_ci_low, mean_ci_high=mean_ci_high, mcse=grid_error , grid_ci_high=grid_ci_high, grid_ci_low=grid_ci_low, n_eff=n_eff, t_multiplier=t_vals)
CSV.write(joinpath(exp, "bathy_statistics_$(burnin)_$(sample_number).csv"), result_df)

bathy_label = latexstring("b_i, \\ \\mathrm{NRMSE} = $(round(bathy_nrmse, digits=3))")

error_plot = scatter(xs, mean_bathy, yerror=grid_error, label="Mean of last $(size(bathy)[1]-burnin) samples",
     ylims=(-0.055,0.21), xlabel=L"x [m]", ylabel=L"b(x) [m]", title="Bathymetry Sample Mean with MCSE Error Bars", grid=true)
plot!(error_plot, xs, mean_bathy; label=bathy_label, color=Plots.palette(:default)[1])
plot!(error_plot, xs, exact_b, label="True Bathymetry", color=:black)

if include_adjoint
    adjoint_bathy = CSV.read(adjoint_file, DataFrame)
    x_adj = adjoint_bathy.x
    bathy_adj = adjoint_bathy.b
    plot!(error_plot, x_adj, bathy_adj, label=adjoint_label, color=:black, linestyle=:dash)
    appendix = "_adjoint"
end

scatter!(error_plot, [3.5,5.5,7.5], [0,0,0], label="Sensor locations", color=:black, markersize=6, marker=:star5)
savefig(error_plot, exp*"/plots/mean_bathy_errorbars_$(burnin)_$(sample_number)$(appendix).png")
savefig(error_plot, exp*"/plots/mean_bathy_errorbars_$(burnin)_$(sample_number)$(appendix).pdf")
println("Store at $(exp*"/plots/mean_bathy_errorbars_$(burnin)_$(sample_number)$(appendix).png")")

ciplot = plot(xs, exact_b; label="Exact bathymetry", color=:black)
plot!(ciplot, xs, mean_bathy, ribbon=(mean_bathy .- grid_ci_low, grid_ci_high .- mean_bathy),  color=Plots.palette(:default)[1], label="95% HPD",
    ylims=(-0.05,0.21), xlabel=L"x \ [m]", ylabel=L"b(x) \ [m]", grid=true)
plot!(ciplot, xs, mean_bathy; label=bathy_label, color=Plots.palette(:default)[2])

if include_adjoint
    adjoint_bathy = CSV.read(adjoint_file, DataFrame)
    x_adj = adjoint_bathy.x
    bathy_adj = adjoint_bathy.b
    plot!(ciplot, x_adj, bathy_adj, label=adjoint_label, color=:black, linestyle=:dash)
    appendix = "_adjoint"
end

scatter!(ciplot, [3.5,5.5,7.5], [0,0,0], label="Sensor locations", color=:black, markersize=6, marker=:star5)
savefig(ciplot, exp*"/plots/mean_bathy_credible_interval_$(burnin)_$(sample_number)$(appendix).png")
savefig(ciplot, exp*"/plots/mean_bathy_credible_interval_$(burnin)_$(sample_number)$(appendix).pdf")
println("Store at $(exp*"/plots/mean_bathy_credible_interval_$(burnin)_$(sample_number)$(appendix).png")")

# Combined comparison plot: HPD vs Mean ± t·MCSE
comp_plot = plot(xs, exact_b; label="Exact bathymetry", color=:black, legend_title="NRMSE = $(round(bathy_nrmse, digits=3))", legend_title_font_pointsize=10)
plot!(comp_plot, xs, mean_bathy, ribbon=(mean_bathy .- grid_ci_low, grid_ci_high .- mean_bathy), color=Plots.palette(:default)[1], label="68% HPD", fillalpha=0.5,
ylims=(-0.05,0.21), xlabel=L"x \ [\mathrm{m}]", ylabel=L"b(x) \ [\mathrm{m}]", grid=true)
plot!(comp_plot, xs, mean_bathy, ribbon=(mean_bathy .- mean_ci_low, mean_ci_high .- mean_bathy), color=Plots.palette(:default)[2], label="Mean ± 95% CI", fillalpha=0.5)
scatter!(comp_plot, [3.5,5.5,7.5], [0,0,0], label="Sensor locations", color=:black, markersize=6, marker=:star5)
# add sparse errorbars for Mean ± t·MCSE on combined plot (every 5th point)
idx = 1:5:length(xs)
#scatter!(comp_plot, xs[idx], mean_bathy[idx], yerror=(t_vals[idx] .* grid_error[idx]), label="Mean ± t·MCSE (points)", color=Plots.palette(:default)[3], marker=:circle)
savefig(comp_plot, exp*"/plots/mean_bathy_hpd_vs_meanci_$(burnin)_$(sample_number)$(appendix).png")
savefig(comp_plot, exp*"/plots/mean_bathy_hpd_vs_meanci_$(burnin)_$(sample_number)$(appendix).pdf")
if out_dir != ""
    title = ""
    file_name = obs_config.real_data ? "reconstruction_$(stepsize)" : "reconstruction_$(prior)_prior_$(proposal_algo)_$(proposal)"
    plot!(comp_plot, legend=false)
    if !obs_config.real_data
        if proposal != "normal"
            plot!(comp_plot, ylabel="", yformatter=:none)
        end
        if prior != "smooth_sparse"
            plot!(comp_plot, xlabel="", xformatter=:none)
        end

    end
    if obs_config.real_data
        title = L"\gamma ="*"$(stepsize)"
        if stepsize in [0.002,0.004]
            plot!(comp_plot, ylabel="", yformatter=:none)
        end
        if stepsize in [0.001,0.002]
            plot!(comp_plot, xlabel="", xformatter=:none)
        end
    end
    savefig(comp_plot, joinpath(out_dir, "$(file_name).png"))
    savefig(comp_plot, joinpath(out_dir, "$(file_name).pdf"))
end

println("Store at $(exp*"/plots/mean_bathy_hpd_vs_meanci_$(burnin)_$(sample_number)$(appendix).png")")

sim_chain = forward(mean_bathy)

if plot_sensor_simulation
    println("#############################\nCreate sensor simulation plots" )
    rel_l2_sim_error = round.(sqrt.(sum((sim_chain .- obs_data.H).^2, dims=1)) ./ sqrt.(sum((obs_data.H).^2, dims=1)), digits=4).*100
    for i in 2:4
        psim = plot(obs_data.t, obs_data.H[:,i-1]; title="Sensor $i, ε=$(rel_l2_sim_error[i-1])%", label="measurement", xlabel=L"t [s]", ylabel=L"H [m]", linestyle=:dash)
        plot!(psim, obs_data.t, sim_chain[:,i-1]; label="simulation ", linestyle=:dot, linewidth=2)
        savefig(psim, joinpath(exp, "plots/sim_chain_sensor_$(i)_$(sample_number).png"))
        savefig(psim, joinpath(exp, "plots/sim_chain_sensor_$(i)_$(sample_number).pdf"))
    end
end


# plot chains
pchain = plot(; xlabel="Iteration", ylabel=L"b(x) \ [\mathrm{m}]", ylims=(-0.1,0.21))
plot!(pchain, bathy_before_burnin)
savefig(pchain, exp*"/plots/chain_$(sample_number).png")
savefig(pchain, exp*"/plots/chain_$(sample_number).pdf")

# plot log posterior, log likelihood, log prior, acceptance rate
pposterior = plot(; xlabel="Iteration", ylabel=L"\log \pi_L(H|b) + \log \pi_\mathrm{Prior}(b)")
plot!(pposterior, lp_before_burnin)
plot!(pposterior, yaxis=(formatter=y->string(round(y / 10^4, digits=2))))
ymin, ymax = ylims(pposterior)
xmin = first(xlims(pposterior))
annotate!(pposterior, [(xmin, ymax + 0.05 * (ymax - ymin), Plots.text(L"\times10^{4}", 8, :black, :center))])
savefig(pposterior, exp*"/plots/log_posterior_$(sample_number).png")
savefig(pposterior, exp*"/plots/log_posterior_$(sample_number).pdf")

plikelihood = plot(; xlabel="Iteration", ylabel=L"\log \pi_L(H|b)")
plot!(plikelihood, ll_before_burnin)
plot!(plikelihood, yaxis=(formatter=y->string(round(y / 10^4, digits=2))))
ymin, ymax = ylims(plikelihood)
xmin = first(xlims(plikelihood))
annotate!(plikelihood, [(xmin, ymax + 0.05 * (ymax - ymin), Plots.text(L"\times10^{4}", 8, :black, :center))])
savefig(plikelihood, exp*"/plots/log_likelihood_$(sample_number).png")
savefig(plikelihood, exp*"/plots/log_likelihood_$(sample_number).pdf")

pprior = plot(; xlabel="Iteration", ylabel=L"\log \pi_\mathrm{Prior}(b)")
plot!(pprior, lprior_before_burnin)
plot!(pprior, yaxis=(formatter=y->string(round(y / 10^2, digits=2))))
ymin, ymax = ylims(pprior)
xmin = first(xlims(pprior))
annotate!(pprior, [(xmin, ymax + 0.05 * (ymax - ymin), Plots.text(L"\times10^{2}", 8, :black, :center))])
savefig(pprior, exp*"/plots/log_prior_$(sample_number).png")
savefig(pprior, exp*"/plots/log_prior_$(sample_number).pdf")

par = plot(; xlabel="Iteration", ylabel="Acceptance Rate")
plot!(par, ar_before_burnin)
savefig(par, exp*"/plots/acceptance_rate_$(sample_number).png")
savefig(par, exp*"/plots/acceptance_rate_$(sample_number).pdf")

println("peak height exp $(exp_bathymetry([4.0]))")
mean_b_interp = PCHIPInterpolation(mean_bathy, xs; extrapolation=ExtrapolationType.Constant)
metrics_dict = Dict("1-NRMSE" => bathy_nrmse,
                    "2-rL2" => bathy_l2,
                    "3-rLinf" => bathy_linf,
                    "4-peak" => mean_b_interp(4.0),
                    "5-max" => maximum(mean_bathy),
                    "6-max_loc" => xs[argmax(mean_bathy)])

metrics_df = DataFrame(metrics_dict)
metrics_file = joinpath(exp, "metrics_$(burnin)_$(sample_number).csv")
CSV.write(metrics_file, metrics_df)

if out_dir != ""
    file_name = obs_config.real_data ? "metrics_$(stepsize)" : "metrics_$(prior)_prior_$(proposal_algo)_$(proposal).csv"
    println("Store metrics at $(joinpath(out_dir, "metrics__$(prior)_prior_$(proposal_algo)_$(proposal).csv"))")
    CSV.write(joinpath(out_dir, file_name), metrics_df)
end