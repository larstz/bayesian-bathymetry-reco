using Pkg
Pkg.activate(".")
using Serialization
using Plots
using StatsPlots
using Statistics
using BathymetryReco
using MCMCChains
using LaTeXStrings
using CSV
using DataFrames

include("my_theme.jl")
theme(:custom)

println("#############################\nRead in chain" )

exp = ARGS[1]
ani = false
chain = deserialize(joinpath(exp, "chain_1.jls"))

config = load_config(joinpath(exp, "experiment_config.toml"))
sim_config = config.sim_params
mcmc_config = config.mcmc_params
obs_config = config.obs_settings

# Load the data
if obs_config.real_data
    obs_data = load_observation(obs_config.path, sim_config.tstart, sim_config.tinterval)
else
    obs_data, exact_b = load_observation(obs_config.path, obs_config.noise_var, sensor_rate=obs_config.sensor_rate)
end

solver = swe_solver(sim_config)
forward(params) = simulation(params, solver, obs_data)

burnin = 1000

bathy = chain[burnin+1:end,1:mcmc_config.dim]
lp = chain[burnin+1:end,mcmc_config.dim+1]
ar = chain[burnin+1:end,mcmc_config.dim+2]

xs = range(1.5,15.0,mcmc_config.dim)
exact_b = exp_bathymetry(xs)

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
bathy_nrmse = sqrt.(mean((mean_bathy .- exact_b).^2)) ./ (maximum(exact_b) .- minimum(exact_b))*100
bathy_l2 = sqrt(sum((mean_bathy .- exact_b).^2))/sqrt(sum(exact_b.^2))*100
bathy_linf = maximum(abs.(mean_bathy .- exact_b))/maximum(exact_b)*100
mcmc_chain = Chains(bathy)
grid_error = mcse(mcmc_chain)[:, :mcse]
grid_ci_low = hpd(mcmc_chain)[:, :lower]
grid_ci_high = hpd(mcmc_chain)[:, :upper]

result_df = DataFrame(x=xs, mean_bathy=mean_bathy, ci_low=grid_ci_low, ci_high=grid_ci_high, mcse=grid_error)
CSV.write(joinpath(exp, "bathy_statistics_$(burnin).csv"), result_df)

bathy_label = L"\mathrm{NRMSE} = %$(round(bathy_nrmse, digits=3))\%"# \\ \mathrm{L}_2 = %$(round(bathy_l2, digits=3))\% \\ \mathrm{L}_\infty = %$(round(bathy_linf, digits=3))\%"

error_plot = scatter(xs, mean_bathy, yerror=grid_error, label="Mean of last $(size(bathy)[1]-burnin) samples",
     ylims=(-0.055,0.21), xlabel=L"x [m]", ylabel=L"b(x) [m]", title="Bathymetry Sample Mean with MCSE Error Bars", grid=true)
plot!(error_plot, xs, mean_bathy; label=bathy_label)
plot!(error_plot, xs, exact_b, label="True Bathymetry", color=:black)
scatter!([3.5,5.5,7.5], [0,0,0], label="Sensor locations", color=:black, markersize=6, marker=:star5)
savefig(error_plot, exp*"/plots/mean_bathy_errorbars_$(burnin).png")
savefig(error_plot, exp*"/plots/mean_bathy_errorbars_$(burnin).pdf")
println("Store at $(exp*"/plots/mean_bathy_errorbars.png")")

ciplot = plot(xs, mean_bathy, ribbon=(mean_bathy .- grid_ci_low, grid_ci_high .- mean_bathy), label="95% Credible Interval",
    ylims=(-0.05,0.21), xlabel=L"x [m]", ylabel=L"b(x) [m]", title="Bathymetry Sample Mean with 95% Credible Interval", grid=true)
plot!(xs, mean_bathy; label=bathy_label)
plot!(ciplot, xs, exact_b; label="Exact bathymetry", color=:black)
savefig(ciplot, exp*"/plots/mean_bathy_credible_interval_$(burnin).png")
savefig(ciplot, exp*"/plots/mean_bathy_credible_interval_$(burnin).pdf")
println("Store at $(exp*"/plots/mean_bathy_credible_interval.png")")

sim_chain = forward(mean_bathy)

println("#############################\nCreate sensor simulation plots" )
rel_l2_sim_error = round.(sqrt.(sum((sim_chain .- obs_data.H).^2, dims=1)) ./ sqrt.(sum((obs_data.H).^2, dims=1)), digits=4).*100
for i in 2:4
    psim = plot(obs_data.t, obs_data.H[:,i-1]; title="Sensor $i, ε=$(rel_l2_sim_error[i-1])%", label="measurement", xlabel=L"t [s]", ylabel=L"H [m]", linestyle=:dash)
    plot!(psim, obs_data.t, sim_chain[:,i-1]; label="simulation ", linestyle=:dot, linewidth=2)
    savefig(psim, joinpath(exp, "plots/sim_chain_sensor_$(i).png"))
    savefig(psim, joinpath(exp, "plots/sim_chain_sensor_$(i).pdf"))
end

metrics_dict = Dict("NRMSE" => bathy_nrmse,
                    "rL2" => bathy_l2,
                    "rLinf" => bathy_linf)
metrics_df = DataFrame(metrics_dict)
metrics_file = joinpath(exp, "metrics.csv")
CSV.write(metrics_file, metrics_df)