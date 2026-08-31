using Pkg
Pkg.activate(".")
using Serialization
using Plots
using StatsPlots
using Statistics
using LaTeXStrings
using MCMCChains
using Distributions
using BathymetryReco


include("my_theme.jl")
theme(:custom)

println("#############################\nRead in chain" )

exp = "data/results/tmcmc/discretized/tmcmc_results_1k/mean_heat_wb_tmcmc_32p" #ARGS[1]

config = load_config(joinpath(exp, "experiment_config.toml"))
sim_config = config.sim_params
obs_config = config.obs_settings

# Load the data
if obs_config.real_data
    obs_data = load_observation(obs_config.path, sim_config.tstart, sim_config.tinterval)
else
    obs_data, exact_b = load_toy_observation(obs_config.path, obs_config.noise_var, sensor_rate=obs_config.sensor_rate)
end

obs_H = reshape(obs_data.H, (1, size(obs_data.H)...))


sensor_simulations = deserialize(joinpath(exp, "sensor_simulations.jls"))
sim_chains = deserialize(joinpath(exp, "sim_chains.jls"))

mean_pred = dropdims(mean(sensor_simulations, dims=1), dims=1)

rel_l2_sim_error = dropdims(sqrt.(sum((sensor_simulations .- obs_H).^2, dims=2)) ./ sqrt.(sum((obs_H).^2, dims=2)), dims=2)

mean_rel_l2_sim_error = round.(mean(rel_l2_sim_error, dims=1).*100, digits=4)
std_rel_l2_sim_error = round.((std(rel_l2_sim_error, dims=1))./sqrt(size(rel_l2_sim_error, 1)).*100, digits=4)

for i in 1:3
    error_string = latexstring("\\varepsilon = $(mean_rel_l2_sim_error[i]) \\pm $(std_rel_l2_sim_error[i])\\%")

    sensor_chain = sim_chains[:,:,i]
    n_eff = ess(sensor_chain)[:, :ess]
    se = mcse(sensor_chain)[:, :mcse]
    t_d = [quantile(TDist(n_e - 1), 0.975) for n_e in n_eff]
    se = se .* t_d
    ci_low = hpd(sensor_chain)[:, :lower]
    ci_high = hpd(sensor_chain)[:, :upper]

    ribbon = (mean_pred[:,i] .- ci_low, ci_high .- mean_pred[:,i])
    ppc_plot = plot(;xlabel= L"t\, [\mathrm{s}]", ylabel= L"H\, [\mathrm{m}]")
    plot!(ppc_plot, obs_data.t, obs_data.H[:,i]; title="Sensor $(i+1), "*error_string, label="measurement", linestyle=:dash)
    plot!(ppc_plot, obs_data.t, mean_pred[:,i]; ribbon=ribbon, label="mean prediction with 95% CI", linestyle=:dot, linewidth=2)
    #scatter!(ppc_plot, obs_data.t, mean_pred[:,i], yerror=se, alpha=0.3, color=:blue)
    savefig(ppc_plot, joinpath(exp, "plots/MCMC_sim_chain_sensor_$(i+1)_ci.png"))
    savefig(ppc_plot, joinpath(exp, "plots/MCMC_sim_chain_sensor_$(i+1)_ci.pdf"))
end
