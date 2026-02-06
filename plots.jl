using Pkg
Pkg.activate(".")

using BathymetryReco
using Plots
using Statistics
using Serialization
using Dates
using LaTeXStrings
using Measures
using Printf
using MCMCChains
using CSV
using DataFrames

include("my_theme.jl")
theme(:custom)
ticks1dec(x) = @sprintf("%.1f", x)
#default(size=(0.25*0.75*600, 0.25*400))
date_pattern = r"(\d{4}-\d{2}-\d{2})"

experiment = ARGS[1]
exp_date = Date(match(date_pattern, experiment).match, DateFormat("Y-mm-dd"))
println("Creating plots for experiment: ", experiment)
config = load_config(joinpath(experiment, "experiment_config.toml"))
sim_config = config.sim_params
mcmc_config = config.mcmc_params
obs_config = config.obs_settings
io_config = config.io_settings

plot_chains = true
plot_bathys = true

# Load the data
if obs_config.real_data
    obs_data = load_observation(obs_config.path, sim_config.tstart, sim_config.tinterval)
    x = Vector(LinRange(sim_config.xbounds[1], sim_config.xbounds[2], sim_config.nx))
    exact_b = exp_bathymetry(x)
else
    obs_data, exact_b = load_observation(obs_config.path, obs_config.noise_var, sensor_rate=obs_config.sensor_rate)
    x = obs_data.sim_x
end

# Load the MCMC samples
files = readdir(experiment)
chains = filter(x -> occursin(r"chain_[0-9]+.jls", x), files)
n_chains = length(chains)
samples = [deserialize(joinpath(experiment, file)) for file in chains]
burn_in = 200

# sampled 2σ² instead of σ², so we need to convert it, adjusted after 2025-06-25
if exp_date < Date(2025,6,25)
    setindex!.(samples, 0.5 .* getindex.(samples, :, 2), :,2)
end

sample_mean = mean.(samples, dims=1)
samples_mat = hcat(samples...)

stored_vals = Int64(size(samples_mat, 2)/n_chains)
param_names = ["\\mu", "\\sigma^2", "lpost", "ll", "lprior","ar"]
chain_param = [:mu, :sigma2, :lpost, :ll, :lprior, :ar]
chain_internals = [:lpost, :ll, :lprior, :ar]
chain_array = reshape(samples_mat[250+1:end, :], (:, stored_vals, n_chains))

mcmc_chains = Chains(chain_array, chain_param, Dict(:internals => chain_internals))

# everything should be loaded now create the plots
plot_path = joinpath(experiment, "plots")
mkpath.(joinpath.(plot_path, ["pdfs", "pngs"]))

println("Plot the MCMC samples")

if plot_chains
    # Plot the chains
    for i in 1:stored_vals
        param = param_names[i]
        param_per_chain = getindex.(samples, :, i)
        pc = plot(;xlabel="Iteration", legend=:outerright)
        plot!(pc, param_per_chain, label=permutedims(latexstring.("$(param)_".*string.(1:n_chains))))
        xaxis!(pc, xminorgrid=true, xlims=(-10,2100))
        if param == "\\sigma^2"
            yaxis!(pc, yticks=0:0.1:1.0, yformatter=ticks1dec, yminorticks=2, ylabel=latexstring("$(param)\\ [\\mathrm{m}^2]"), yminorgrid=true)
        end
        if param == "\\mu"
            yaxis!(pc, yticks=0.0:1.0:7.0, ylim=(0.9,7.1),yformatter=ticks1dec, ylabel=latexstring("$(param)\\ [\\mathrm{m}]"))
        end
        savefig(pc, joinpath(plot_path, "pngs", "chain_val_$(i).png"))
        savefig(pc, joinpath(plot_path, "pdfs", "chain_val_$(i).pdf"))
    end
end
println("Plot the bathymetry samples and sensor simulations per bathymetry")
# Plot Bathymetries and sensor simulations
if plot_bathys
    for (idx, chain) in enumerate(samples)
        # Plot the bathymetry
        burnin_chain = chain[burn_in+1:end, :]
        bathys = zeros(size(burnin_chain)[1], length(exact_b))
        for (i, sample) in enumerate(eachrow(burnin_chain))
            bathys[i, :] = bathymetry(x, sample[1:end-4])
        end
        bathy_mean = vec(mean(bathys, dims=1))
        mean_params = vec(mean(burnin_chain[:, 1:end-4], dims=1))
        rel_l2_error = round.(sqrt(sum((bathymetry(x, mean_params) .- exact_b).^2)) ./ sqrt(sum((exact_b).^2))*100, digits=2)

        pb = plot(x, exact_b; c=:black, xlabel="x [m]", ylabel="b(x) [m]", label="Exact bathymetry", xlim=(1.5,12))
        plot!(pb, x, bathys'; label=permutedims(vcat([latexstring("b(x; \\mu_i, \\sigma^2_i),\\ i \\in [n]")], repeat([""], size(bathys)[1]))), alpha=0.1, lw=0.25, color=:gray)
        #plot!(pb, x, bathy_mean; c=:red, label="sample mean, ε=$(rel_l2_error_mean)")
        plot!(pb, x, bathymetry(x, mean_params); c=:blue, label=latexstring("b(x; \\bar{\\mu},\\bar{\\sigma}^2),\\ \\varepsilon_\\mathrm{L2}=$(rel_l2_error)\\%"))
        yaxis!(pb, yticks=0.0:0.05:0.5)
        xaxis!(pb, xticks=2:2:12)
        # to display error use bars from quantiles

        savefig(pb, joinpath(plot_path, "pngs", "bathy_chain_$(idx)_bi_$(burn_in).png"))
        savefig(pb, joinpath(plot_path, "pdfs", "bathy_chain_$(idx)_bi_$(burn_in).pdf"))

        mcmc_bathy = Chains(bathys)
        ci_low = hpd(mcmc_bathy)[:, :lower]
        ci_high = hpd(mcmc_bathy)[:, :upper]
        bathy_l2 = round.(sqrt(sum((bathy_mean .- exact_b).^2)) / sqrt(sum((exact_b).^2)), digits=4)
        bathy_linf = maximum(abs.(bathy_mean .- exact_b))/maximum(exact_b)*100
        bathy_nrmse = (sqrt(mean(bathy_mean .- exact_b).^2) / (maximum(exact_b) - minimum(exact_b)))*100

        metrics_dict = Dict("NRMSE" => bathy_nrmse,
                    "rL2" => bathy_l2,
                    "rLinf" => bathy_linf)
        metrics_df = DataFrame(metrics_dict)
        metrics_file = joinpath(experiment, "metrics_$(idx)_$(burn_in).csv")
        CSV.write(metrics_file, metrics_df)

        ciplot = plot(x, exact_b; label="Exact bathymetry", color=:black)
        plot!(ciplot, x, bathy_mean, ribbon=(bathy_mean .- ci_low, ci_high .- bathy_mean),  color=Plots.palette(:default)[1], label="95% Credible Interval",
        ylims=(-0.01,0.21), xlabel="x [m]", ylabel="b(x) [m]", title="Bathymetry Sample Mean with 95% Credible Interval", grid=true)
        plot!(ciplot, x, bathy_mean;   color=Plots.palette(:default)[2], label=latexstring("\\bar{b}(x;\\hat{b}^{(i)}_p, \\hat{b}^{(i)}_w) \\ \\mathrm{NRMSE} = $(round(bathy_nrmse, digits=3))"))
        savefig(ciplot, joinpath(plot_path, "pngs", "mean_bathy_credible_interval_$(idx)_bi_$(burn_in).png"))
        savefig(ciplot, joinpath(plot_path, "pdfs", "mean_bathy_credible_interval_$(idx)_bi_$(burn_in).pdf"))

        # Plot the sensor simulation
        sim_chain = simulation(mean_params, sim_config, obs_data)

        rel_l2_sim_error = round.(sqrt.(sum((sim_chain .- obs_data.H).^2, dims=1)) ./ sqrt.(sum((obs_data.H).^2, dims=1)), digits=4)
        for i in 2:4
            psim = plot(obs_data.t, obs_data.H[:,i-1]; title="Sensor $i, ε=$(rel_l2_sim_error[i-1])", label="measurement", xlabel="t [s]", ylabel="H [m]", linestyle=:dash)
            plot!(psim, obs_data.t, sim_chain[:,i-1]; label="simulation ", linestyle=:dot, linewidth=2)
            savefig(psim, joinpath(plot_path, "pngs", "sim_chain_$(idx)_sensor_$(i).png"))
            savefig(psim, joinpath(plot_path, "pdfs", "sim_chain_$(idx)_sensor_$(i).pdf"))
        end

    end
end
