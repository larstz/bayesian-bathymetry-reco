using Pkg
Pkg.activate(".")
using Serialization
using Plots
using StatsPlots
using Statistics
using LaTeXStrings

using Distributed
addprocs(7)
@everywhere begin
    using BathymetryReco
end



include("my_theme.jl")
theme(:custom_slides)

println("#############################\nRead in chain" )

exp = "./data/results/results_for_paper/2026-02-09-13-53-04_mean_heat_wb" #ARGS[1]
ani = false
include_adjoint = false
plot_sensor_simulation = false
appendix = ""
chain = deserialize(joinpath(exp, "chain_1.jls"))

config = load_config(joinpath(exp, "experiment_config.toml"))
sim_config = config.sim_params
mcmc_config = config.mcmc_params
obs_config = config.obs_settings

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


@everywhere forward(params) = simulation(params, $sim_config, $obs_data)

burnin = 1000
sample_number = 4000

bathy = chain[burnin+1:burnin+sample_number,1:mcmc_config.dim]
lp = chain[burnin+1:burnin+sample_number,mcmc_config.dim+1]
ar = chain[burnin+1:burnin+sample_number,end]

sensor_simulations = Array{Float64}(undef, length(obs_data.t), length(obs_data.x), sample_number)

for i in 1:sample_number
    println("Simulating for sample ", i, " with log-posterior ", lp[i], " and acceptance rate ", ar[i])
    sim = forward(bathy[i, :])
    sensor_simulations[:, :, i] = sim
end


mean_pred = dropdims(mean(sensor_simulations, dims=3), dims=3)

# 95% confidence interval (2.5% and 97.5%) across the sample dimension (3rd dim)
# Resulting `q_low` and `q_high` are matrices with dimensions
# (ntime, nposition) matching `mean_pred`.
q_low = [quantile(view(sensor_simulations, t, p, :), 0.025) for t in 1:size(sensor_simulations,1), p in 1:size(sensor_simulations,2)]
q_high = [quantile(view(sensor_simulations, t, p, :), 0.975) for t in 1:size(sensor_simulations,1), p in 1:size(sensor_simulations,2)]

rel_l2_sim_error = sqrt.(sum((sensor_simulations .- obs_data.H).^2, dims=1)) ./ sqrt.(sum((obs_data.H).^2, dims=1))

mean_rel_l2_sim_error = dropdims(round.(mean(rel_l2_sim_error, dims=3).*100, digits=4), dims=3)
std_rel_l2_sim_error = dropdims(round.(1.96.*(std(rel_l2_sim_error, dims=3))./sqrt(sample_number).*100, digits=4), dims=3)

for i in 1:3
    ribbon = (mean_pred[:,1] .- q_low[:,1], q_high[:,1] .- mean_pred[:,1])
    ppc_plot = plot(;title="Posterior Predictive Check", xlabel="time [s]", ylabel="Water surface height [m]")
    plot!(ppc_plot, obs_data.t, obs_data.H; title="Sensor $i, ε=$(mean_rel_l2_sim_error[i]) +-$(std_rel_l2_sim_error[i])%", label="measurement", xlabel=L"t [s]", ylabel=L"H [m]", linestyle=:dash)
    plot!(ppc_plot, obs_data.t, mean_pred[:,1]; ribbon=ribbon, label="mean prediction with 95% CI", linestyle=:dot, linewidth=2)
    savefig(ppc_plot, joinpath(exp, "plots/sim_chain_sensor_$(i)_ci.png"))
    savefig(ppc_plot, joinpath(exp, "plots/sim_chain_sensor_$(i)_ci.pdf"))
end