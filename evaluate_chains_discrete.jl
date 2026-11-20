using Pkg
Pkg.activate(".")

using BathymetryReco
using DataFrames
using Statistics
using Serialization
using Dates
using CSV
using LaTeXStrings
using MCMCChains
using Plots
using Distributions
using DataInterpolations

include("my_theme.jl")
theme(:custom)

function nrmse(exp_b, est)
    return sqrt(mean((exp_b .- est).^2))/(maximum(exp_b) - minimum(exp_b))
end

show_plots = false
experiment = "data/results/paper_results/chains/heat_tests/mean_tests/sensor-2-3-4/prior-smooth-sparse/proposal-rw-smooth/stepsize-0.002/100k_samples/"
chain_list = String[]
configs = String[]
for (exp,_, files) in walkdir(experiment)
    if rstrip(splitdir(exp)[1], '/') == rstrip(experiment, '/')
        append!(chain_list,joinpath.(exp,filter(x -> occursin(r"chain_[0-9]+.jls", x), files)))
    end
end

# Load the MCMC samples
chains = filter(x -> occursin(r"chain_[0-9]+.jls", x), chain_list)

samples = deserialize.(chains)

chain_tensor = reshape(hcat(samples...), size(samples[1])[1], size(samples[1])[2], size(samples)[1])
parameter_start = 1
parameter_end = 64
chain_tensor = chain_tensor[:, parameter_start:parameter_end, :]
initial = chain_tensor[1, :, :]
burn_in = 1000
chain_tensor = chain_tensor[burn_in:end, :, :]

xs = collect(LinRange(1.5, 15.0, parameter_end - parameter_start + 1))

if show_plots
    init_plot = plot(xs, initial, label=reshape(["Chain $i" for i in 1:size(initial, 2)], 1,:), xlabel=L"x \ [\mathrm{m}]", ylabel=L"b(x) \ [\mathrm{m}]", title="Initial Samples")

    savefig(init_plot, experiment*"/bathy_initial.png")
    savefig(init_plot, experiment*"/bathy_initial.pdf")
end

mcmc_chains = Chains(chain_tensor)
means = mean(mcmc_chains, append_chains=false)
hpds = hpd(mcmc_chains, append_chains=false)
rh = rhat(mcmc_chains, kind=:basic)[:,:rhat]
println("R-hat values: ", sum(rh .> 1.1), " out of ", length(rh), " chains have R-hat > 1.1")

mean_bathys = [mb[:,:mean] for mb in means]
mean_bathys = hcat(mean_bathys...)

# summary statistics for the mean bathymetry
exact_b = exp_bathymetry(xs)
bathy_nrmse = sqrt.(mean((mean_bathys .- exact_b).^2, dims=1)) ./ (maximum(exact_b) .- minimum(exact_b))*100
bathy_l2 = sqrt.(sum((mean_bathys .- exact_b).^2, dims=1))/sqrt(sum(exact_b.^2))*100
bathy_linf = maximum(abs.(mean_bathys .- exact_b), dims=1)/maximum(exact_b)*100

# qualitative metrics for the mean bathymetry
mean_b_interp = [PCHIPInterpolation(mean_bathy, xs; extrapolation=ExtrapolationType.Constant) for mean_bathy in eachcol(mean_bathys)]
peak_height = [mean_b_interp[i](4.0) for i in 1:size(mean_bathys, 2)]
max_height = vec(maximum(mean_bathys, dims=1))
max_pos = xs[[argmax(mean_bathy) for mean_bathy in eachcol(mean_bathys)]]
max_id_13 = mean_bathys[13, :]

metric_names = ["1-NRMSE", "2-rL2", "3-rLinf", "4-peak", "5-max", "6-max_loc", "7-max id 13"]
metric_values = [vec(bathy_nrmse), vec(bathy_l2), vec(bathy_linf), peak_height, max_height, max_pos, vec(max_id_13)]
metrics_df = DataFrame(metric=metric_names,
                       mean=[mean(values) for values in metric_values],
                       se=[std(values)/sqrt(length(values)) for values in metric_values])
CSV.write(joinpath(experiment, "summary_metrics.csv"), metrics_df)
display(metrics_df)


mean_difference = maximum(mean_bathys,dims=2) - minimum(mean_bathys,dims=2)

hpd_upper = [hpd[:,:upper] for hpd in hpds]
hpd_upper = hcat(hpd_upper...)
hpd_lower = [hpd[:,:lower] for hpd in hpds]
hpd_lower = hcat(hpd_lower...)
hpd_widths = hpd_upper - hpd_lower
mean_hpd_width = vec(mean(hpd_widths, dims=2))

# Compare chain means relative to posterior uncertainty.
mean_difference = vec(mean_difference)
relative_mean_difference = mean_difference ./ mean_hpd_width

# Compare the uncertainty estimates from the individual chains.
hpd_width_difference = vec(maximum(hpd_widths, dims=2) - minimum(hpd_widths, dims=2))
relative_hpd_width_difference = hpd_width_difference ./ mean_hpd_width
hpd_intersection = max.(0.0, vec(minimum(hpd_upper, dims=2) - maximum(hpd_lower, dims=2)))
hpd_union = vec(maximum(hpd_upper, dims=2) - minimum(hpd_lower, dims=2))
hpd_overlap = hpd_intersection ./ hpd_union

comparison_df = DataFrame(
    metric=["mean disagreement / mean HPD width",
        "HPD width disagreement / mean HPD width",
        "common HPD overlap"],
    mean=[mean(relative_mean_difference),
        mean(relative_hpd_width_difference),
        mean(hpd_overlap)],
    worst=[maximum(relative_mean_difference),
           maximum(relative_hpd_width_difference),
           minimum(hpd_overlap)])
CSV.write(joinpath(experiment, "chain_comparison.csv"), comparison_df)
display(comparison_df)

if show_plots
comp_plot = plot()

for (i, (mb, hpd)) in enumerate(zip(means, hpds))
    mean_bathy = mb[:,:mean]
    grid_ci_low = hpd[:,:lower]
    grid_ci_high = hpd[:,:upper]
plot!(comp_plot, xs, mean_bathy, ribbon=(mean_bathy .- grid_ci_low, grid_ci_high .- mean_bathy), label="95% CI, $i", fillalpha=0.2,
ylims=(-0.05,0.21), xlabel=L"x \ [\mathrm{m}]", ylabel=L"b(x) \ [\mathrm{m}]", grid=true)
end
savefig(comp_plot, experiment*"/bathy_multi_chain_1k.png")
savefig(comp_plot, experiment*"/bathy_multi_chain_1k.pdf")

comp_plot = plot()

for (i, mb) in enumerate(means)
    mean_bathy = mb[:,:mean]
plot!(comp_plot, xs, mean_bathy, label="mean, $i",
ylims=(-0.05,0.21), xlabel=L"x \ [\mathrm{m}]", ylabel=L"b(x) \ [\mathrm{m}]", grid=true)
end
savefig(comp_plot, experiment*"/bathy_multi_chain_means_1k.png")
savefig(comp_plot, experiment*"/bathy_multi_chain_means_1k.pdf")

rh_plot = plot(xs, rh, xlabel=L"x \ [\mathrm{m}]", ylabel=L"\hat{R}")
hline!(rh_plot, [1.1], color=:red, linestyle=:dash)
scatter!(rh_plot,[3.5,5.5,7.5], [1,1,1], label="Sensor locations", color=:black, markersize=6, marker=:star5)
savefig(rh_plot, experiment*"/rhat.png")
savefig(rh_plot, experiment*"/rhat.pdf")
end