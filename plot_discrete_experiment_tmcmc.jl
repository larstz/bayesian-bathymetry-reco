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
using DataInterpolations
using TOML

include("my_theme.jl")
theme(:pamm)

println("#############################\nRead in chain" )

exp = ARGS[1]
#exp = "data/results/results_lars/mcmc_experiment-data_sparse-smooth"

ani = false
plot_sensor_simulation = true
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

solver = swe_solver(sim_config)
forward(params) = simulation(params, solver, obs_data)

burnin = 1000

bathy = chain[burnin+1:5000,1:mcmc_config.dim]
lp = chain[burnin+1:5000,mcmc_config.dim+1]
ar = chain[burnin+1:5000,end]

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

## load TMCMC
using JLD
using TOML
tmcmc_path = ARGS[2]
#tmcmc_path = "data/results/tmcmc/discretized/tmcmc_results_4k/mean_heat_with_tmcmc_65procs/"
tmcmc_samples = load(tmcmc_path * "final_parameters.jld")
bathy_tmcmc = tmcmc_samples["final_parameters"]
tmcmc_config = TOML.parsefile(tmcmc_path * "timings.toml")
tmcmc_config = tmcmc_config["time_summary"]

## mcmc
mean_bathy = vec(mean(bathy, dims=1))
bathy_nrmse = sqrt.(mean((mean_bathy .- exact_b).^2)) ./ (maximum(exact_b) .- minimum(exact_b))*100
bathy_l2 = sqrt(sum((mean_bathy .- exact_b).^2))/sqrt(sum(exact_b.^2))*100
bathy_linf = maximum(abs.(mean_bathy .- exact_b))/maximum(exact_b)*100
mcmc_chain = Chains(bathy)
grid_error = mcse(mcmc_chain)[:, :mcse]
grid_ci_low = hpd(mcmc_chain)[:, :lower]
grid_ci_high = hpd(mcmc_chain)[:, :upper]
# tmcmc
mean_bathy_tmcmc = vec(mean(bathy_tmcmc, dims=1))
bathy_nrmse_tmcmc = sqrt.(mean((mean_bathy_tmcmc .- exact_b).^2)) ./ (maximum(exact_b) .- minimum(exact_b))*100
bathy_l2_tmcmc = sqrt(sum((mean_bathy_tmcmc .- exact_b).^2))/sqrt(sum(exact_b.^2))*100
bathy_linf_tmcmc = maximum(abs.(mean_bathy_tmcmc .- exact_b))/maximum(exact_b)*100
mcmc_chain_tmcmc = Chains(bathy_tmcmc)
grid_error_tmcmc = mcse(mcmc_chain_tmcmc)[:, :mcse]
grid_ci_low_tmcmc = hpd(mcmc_chain_tmcmc)[:, :lower]
grid_ci_high_tmcmc = hpd(mcmc_chain_tmcmc)[:, :upper]

result_df_tmcmc = DataFrame(x=xs, mean_bathy=mean_bathy_tmcmc, ci_low=grid_ci_low_tmcmc, ci_high=grid_ci_high_tmcmc, mcse=grid_error_tmcmc)
CSV.write(joinpath(exp, "bathy_statistics_tmcmc_$(size(bathy_tmcmc,1)).csv"), result_df_tmcmc)

bathy_label = latexstring("b_i, \\ \\mathrm{NRMSE} = $(round(bathy_nrmse, digits=3))")
bathy_label_tmcmc = latexstring("b_i, \\ \\mathrm{NRMSE} = $(round(bathy_nrmse_tmcmc, digits=3))")

error_plot = scatter(xs, mean_bathy, yerror=grid_error, label="Mean $(size(bathy,1)) MH-MCMC samples",
     ylims=(-0.055,0.21), xlabel=L"x [m]", ylabel=L"b(x) [m]", title="Bathymetry Sample Mean with MCSE Error Bars", grid=true)
plot!(error_plot, xs, mean_bathy; label=bathy_label, color=Plots.palette(:default)[1])
scatter!(error_plot, xs, mean_bathy_tmcmc, yerror=grid_error_tmcmc, label="Mean $(size(bathy_tmcmc,1)) TMCMC samples", 
color=Plots.palette(:default)[2])
plot!(error_plot, xs, mean_bathy_tmcmc; label=bathy_label_tmcmc, color=Plots.palette(:default)[2])
plot!(error_plot, xs, exact_b, label="True Bathymetry", color=:black)


scatter!(error_plot, [3.5,5.5,7.5], [0,0,0], label="Sensor locations", color=:black, markersize=6, marker=:star5)
savefig(error_plot, tmcmc_path*"/plots/mean_bathy_errorbars_MCMC$(burnin)_TMCMC$(size(bathy_tmcmc,1)).png")
savefig(error_plot, tmcmc_path*"/plots/mean_bathy_errorbars_MCMC$(burnin)_TMCMC$(size(bathy_tmcmc,1)).pdf")
println("Store at $(tmcmc_path*"/plots/mean_bathy_errorbars.png")")


## ci plot
ciplot = plot(xs, exact_b; label="Exact bathymetry", color=:black, lw=3)
plot!(ciplot, xs, mean_bathy, ribbon=(mean_bathy .- grid_ci_low, grid_ci_high .- mean_bathy),  color=Plots.palette(:default)[1], 
label="MH-MCMC 95% CI", ylims=(-0.05,0.21), xlabel=L"x \ [m]", ylabel=L"b(x) \ [m]", grid=true, lw=2)
plot!(ciplot, xs, mean_bathy; label=bathy_label, color=Plots.palette(:default)[1], lw=2)
plot!(ciplot, xs, mean_bathy_tmcmc, ribbon=(mean_bathy_tmcmc .- grid_ci_low_tmcmc, grid_ci_high_tmcmc .- mean_bathy_tmcmc),  color=Plots.palette(:default)[2], 
label="TMCMC 95% CI", ylims=(-0.05,0.21), xlabel=L"x \ [m]", ylabel=L"b(x) \ [m]", grid=true, lw=1)
plot!(ciplot, xs, mean_bathy_tmcmc; label=bathy_label_tmcmc, color=Plots.palette(:default)[2], lw=1)


scatter!(ciplot, [3.5,5.5,7.5], [0,0,0], label="Sensor locations", color=:black, markersize=6, marker=:star5)
savefig(ciplot, tmcmc_path*"/plots/mean_bathy_credible_interval_MH-MCMC$(burnin)_TMCMC$(size(bathy_tmcmc,1)).png")
savefig(ciplot, tmcmc_path*"/plots/mean_bathy_credible_interval_MH-MCMC$(burnin)_TMCMC$(size(bathy_tmcmc,1)).pdf")
println("Store at $(tmcmc_path*"/plots/mean_bathy_credible_interval_MH-MCMC$(burnin)_TMCMC$(size(bathy_tmcmc,1)).png")")

sim_chain = forward(mean_bathy)
sim_chain_tmcmc = forward(mean_bathy_tmcmc)

if plot_sensor_simulation
    println("#############################\nCreate sensor simulation plots" )
    rel_l2_sim_error = round.(sqrt.(sum((sim_chain .- obs_data.H).^2, dims=1)) ./ sqrt.(sum((obs_data.H).^2, dims=1)), digits=4).*100
    used_ylims = []
    for i in 2:4
        psim = plot(obs_data.t, obs_data.H[:,i-1]; title="Sensor $i, ε=$(rel_l2_sim_error[i-1])%", label="measurement", xlabel=L"t [s]", ylabel=L"H [m]", linestyle=:dash)
        plot!(psim, obs_data.t, sim_chain[:,i-1]; label="simulation ", linestyle=:dot, linewidth=2)
        push!(used_ylims,ylims(psim))
        savefig(psim, joinpath(tmcmc_path, "plots/MCMC_sim_chain_sensor_$(i).png"))
        savefig(psim, joinpath(tmcmc_path, "plots/MCMC_sim_chain_sensor_$(i).pdf"))
    end
    println("#############################\nCreate sensor simulation plots TMCMC" )
    rel_l2_sim_error_tmcmc = round.(sqrt.(sum((sim_chain_tmcmc .- obs_data.H).^2, dims=1)) ./ sqrt.(sum((obs_data.H).^2, dims=1)), digits=4).*100
    for (j,i) in enumerate(2:4)
        psim = plot(obs_data.t, obs_data.H[:,i-1]; title="Sensor $i, ε=$(rel_l2_sim_error_tmcmc[i-1])%", label="measurement", xlabel=L"t [s]", ylabel=L"H [m]", linestyle=:dash)
        plot!(psim, obs_data.t, sim_chain_tmcmc[:,i-1]; label="simulation ", linestyle=:dot, linewidth=2)
        plot!(psim,ylims=used_ylims[j])
        savefig(psim, joinpath(tmcmc_path, "plots/TMCMC_sim_chain_sensor_$(i).png"))
        savefig(psim, joinpath(tmcmc_path, "plots/TMCMC_sim_chain_sensor_$(i).pdf"))
    end
end

println("peak height exp $(exp_bathymetry([4.0]))")
mean_b_interp = PCHIPInterpolation(mean_bathy, xs; extrapolation=ExtrapolationType.Constant)
metrics_dict = Dict("NRMSE" => bathy_nrmse,
                    "rL2" => bathy_l2,
                    "rLinf" => bathy_linf,
                    "peak" => mean_b_interp(4.0))
metrics_df = DataFrame(metrics_dict)
metrics_file = joinpath(exp, "MCMC_metrics_$(burnin).csv")
CSV.write(metrics_file, metrics_df)

println("TMCMC peak height exp $(exp_bathymetry([4.0]))")
mean_b_interp = PCHIPInterpolation(mean_bathy_tmcmc, xs; extrapolation=ExtrapolationType.Constant)
metrics_dict = Dict("NRMSE" => bathy_nrmse_tmcmc,
                    "rL2" => bathy_l2_tmcmc,
                    "rLinf" => bathy_linf_tmcmc,
                    "peak" => mean_b_interp(4.0))
metrics_df = DataFrame(metrics_dict)
metrics_file = joinpath(exp, "TMCMC_metrics_$(size(bathy_tmcmc,1)).csv")
CSV.write(metrics_file, metrics_df)