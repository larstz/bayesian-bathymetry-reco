using Pkg
Pkg.activate(".")
using Serialization
using Plots
using StatsPlots
using Statistics
using LaTeXStrings
using MCMCChains
using Distributions
using Distributed

parallel = false

if parallel
    addprocs(2)
    @everywhere begin
        using BathymetryReco
    end
else
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
else
    obs_data, exact_b = load_toy_observation(obs_config.path, obs_config.noise_var, sensor_rate=obs_config.sensor_rate)
end

obs_H = reshape(obs_data.H, (1, size(obs_data.H)...))

if parallel
    @everywhere forward(params) = simulation(params, $sim_config, $obs_data)
else
    forward(params) = simulation(params, sim_config, obs_data)
end

burnin = 1000
sample_number = size(chain,1) - burnin

bathy = chain[burnin+1:burnin+sample_number,1:mcmc_config.dim]
lp = chain[burnin+1:burnin+sample_number,mcmc_config.dim+1]
ar = chain[burnin+1:burnin+sample_number,end]

sensor_simulations = Array{Float64}(undef, sample_number, length(obs_data.t), length(obs_data.x))

if parallel
    @distributed for i in 1:sample_number
        println("Simulating for sample ", i, " with log-posterior ", lp[i], " and acceptance rate ", ar[i])
        sim = forward(bathy[i, :])
        println(size(sim))
        println(size(sensor_simulations[i, :, :]))
        sensor_simulations[i, :, :] = sim
    end
else
    for i in 1:sample_number
        println("Simulating for sample ", i, " with log-posterior ", lp[i], " and acceptance rate ", ar[i])
        @time sim = forward(bathy[i, :])
        sensor_simulations[i, :, :] = sim
    end
end

mean_pred = dropdims(mean(sensor_simulations, dims=1), dims=1)

sim_chains = Chains(sensor_simulations)

rel_l2_sim_error = dropdims(sqrt.(sum((sensor_simulations .- obs_H).^2, dims=2)) ./ sqrt.(sum((obs_H).^2, dims=2)), dims=2)

mean_rel_l2_sim_error = round.(mean(rel_l2_sim_error, dims=1).*100, digits=4)
std_rel_l2_sim_error = round.((std(rel_l2_sim_error, dims=1))./sqrt(sample_number).*100, digits=4)

for i in 1:3
    error_string = latexstring("\\varepsilon = $(mean_rel_l2_sim_error[i]) \\pm $(std_rel_l2_sim_error[i])\\%")

    sensor_chain = sim_chains[i]
    #ess = ess(sensor_chain)[:, :ess]
    se = mcse(sensor_chain)[:, :mcse]
    t_d = [quantile(TDist(10.- 1), 0.975) for n_e in ess]
    se = se .* t_d
    ci_low = hpd(sensor_chain)[:, :lower]
    ci_high = hpd(sensor_chain)[:, :upper]

    ribbon = (mean_pred[:,i] .- ci_low[:,i], ci_high[:,i] .- mean_pred[:,i])
    ppc_plot = plot(;xlabel= L"t [\mathrm{s}]", ylabel= L"H [\mathrm{m}]")
    plot!(ppc_plot, obs_data.t, obs_data.H[:,i]; title="Sensor $i, ε=$(mean_rel_l2_sim_error[i]) +-$(std_rel_l2_sim_error[i])%", label="measurement", linestyle=:dash)
    plot!(ppc_plot, obs_data.t, mean_pred[:,i]; ribbon=ribbon, label="mean prediction with 95% CI", linestyle=:dot, linewidth=2)
    scatter!(ppc_plot, obs_data.t, mean_pred[:,i], yerror=se, alpha=0.3, color=:blue)
    savefig(ppc_plot, joinpath(exp, "plots/sim_chain_sensor_$(i)_ci.png"))
    savefig(ppc_plot, joinpath(exp, "plots/sim_chain_sensor_$(i)_ci.pdf"))
end