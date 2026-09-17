using Pkg
Pkg.activate(".")

using BathymetryReco
using DataFrames
using Statistics
using Serialization
using Dates
using LaTeXStrings
using MCMCChains
using CairoMakie
using Distributions

function nrmse(p_true, p_est)
    xs = collect(LinRange(1.5, 15.0, 64))
    b_true = bathymetry(xs, p_true)
    b_est = bathymetry(xs, p_est)
    return sqrt(mean((b_true .- b_est).^2))/(maximum(b_true) - minimum(b_true))
end

pt_per_unit = 1
figsize = (390, round(Int, 390*0.618))

# Half size with no whitespace
half_width = figsize[1] / 2
half_height = figsize[2] / 1.5
plot_size = (half_width, half_height)

set_theme!(theme_latexfonts(),
            fontsize=12,
            size=plot_size,
            linewidth=1,
            markersize=4,
            figure_padding=5,
            colormap = :tab10,
            colorrange = (1, 10))

experiment = "data/results/paper_results/height_test/toy_tests/sensor-2-3-4/prior-uniform-uniform-uniform/proposal-rw/stepsize-0.1-0.005-0.01/first"#ARGS[1]
chain_list = String[]
configs = String[]
for (exp,_, files) in walkdir(experiment)
    if rstrip(splitdir(exp)[1], '/') == rstrip(experiment, '/')
        append!(chain_list,joinpath.(exp,filter(x -> occursin(r"chain_[0-9]+.jls", x), files)))
    end
end

# Load the MCMC samples
chains = filter(x -> occursin(r"chain_[0-9]+.jls", x), chain_list)
targets = [match(r"\[.+\]", x).match for x in chains]
targets_vec = unique(Meta.parse.(targets) .|> eval)

mu_targets = [x[1] for x in targets_vec]
s2_targets = [x[2] for x in targets_vec]
h_targets = size(targets_vec[1])[1] < 3 ? [0.2 for x in targets_vec] : [x[3] for x in targets_vec]

n_targets = length(targets_vec)
n_chains = round(Int64, length(chains)/n_targets)

samples = deserialize.(chains)

# adapt to new format also storing loglikelihood and logprior
for (i, sample) in enumerate(samples)
    if size(sample,2) >= 6
        samples[i] = sample[:, [1,2,3,4,end]] # keep only mu, s2, logp, ar
    end
end

chain_tensor = reshape(hcat(samples...), size(samples[1])[1], size(samples[1])[2], n_chains, n_targets)
chain_tensor = chain_tensor[:, 1:4, :, :]
burn_in = 200
chain_tensor = chain_tensor[burn_in:end, :, :, :]
param_names = ["mu", "s2", "h", "lp"]
mean_lp = mean(chain_tensor[:, 4, :, :], dims=1)
d_lp = maximum(mean_lp, dims=2) .- mean_lp

exp_id = [reshape(d_lp[:,:,i].<2, 7) for i in 1:n_targets]

mu_means = zeros(n_targets)
s2_means = zeros(n_targets)
h_means = zeros(n_targets)
mu_error = zeros(n_targets)
s2_error = zeros(n_targets)
h_error = zeros(n_targets)
ci_low = zeros(n_targets, 3)
ci_high = zeros(n_targets, 3)
nrmse_vals = zeros(n_targets)
for (i, id) in enumerate(exp_id)
    mu_means[i] = mean(chain_tensor[:, 1, id, i])
    s2_means[i] = mean(chain_tensor[:, 2, id, i])
    h_means[i] = mean(chain_tensor[:, 3, id, i])
    temp_chain = Chains(chain_tensor[:, :, id, i], param_names, Dict(:internal => [:lp, :ar]))
    error = mcse(temp_chain)
    n_eff = ess(temp_chain)[:, :ess]
    t_d = [quantile(TDist(n_e - 1), 0.975) for n_e in n_eff]
    mu_error[i] = t_d[1] * error[:mu,:mcse]
    s2_error[i] = t_d[2] * error[:s2,:mcse]
    h_error[i] = t_d[3] * error[:h,:mcse]
    ci_low[i, :] = hpd(temp_chain)[:, :lower]
    ci_high[i,:] = hpd(temp_chain)[:, :upper]

    nrmse_vals[i] = nrmse(targets_vec[i], [mu_means[i], s2_means[i], h_means[i]])
end

title = ""
include_title = false
if occursin("width_test", experiment)
    target = s2_targets
    xlabel = L"target $b_w$"
    mutarget = L"target $b_p=4.0$"
    s2target = L"target $b_w$"
    htarget = L"target $h=0.2$"
    extension = "width"
    if include_title
        title = L"Target $b_w$ vs reconstructed $b_w$"
    end
    legend_position = [:rb, :lt]
    ylims_m = (-1, 1)
    ylims_s = (-0.05, 0.25)
    ylims_h = (-0.05, 0.05)
elseif occursin("peak_test", experiment)
    target = mu_targets
    xlabel = L"target $b_p$"
    mutarget = L"target $b_p$"
    s2target = L"target $b_w=0.05$"
    htarget = L"target $h=0.2$"
    extension = "position"
    if include_title
        title = L"Target $b_p$ vs reconstructed $b_p$"
    end
    legend_position = [:rb, :lt]
    ylims_m = (-1, 1)
    ylims_s = (-0.05, 0.25)
    ylims_h = (-0.05, 0.05)
else
    target = h_targets
    xlabel = L"target $b_h$"
    mutarget = L"target $b_p$"
    s2target = L"target $b_w=0.05$"
    htarget = L"target $b_h$"
    extension = "height"
    if include_title
        title = L"Target $b_h$ vs reconstructed $b_h$"
    end
    legend_position = [:rb, :lt]
    ylims_m = (-1, 1)
    ylims_s = (-0.05, 0.25)
    ylims_h = (-0.05, 0.05)
end

res_p = mu_means .- mu_targets
figm = Figure()
axm = Axis(figm[1,1], title=title, xlabel=xlabel, ylabel=L"(reco-target) $b_p$", yticks=-1:0.5:1)
ylims!(axm, ylims_m...)
lines!(axm, target, zeros(length(target)), color = 4, colormap = :tab10, colorrange = (1, 10))
errorbars!(axm, target, (res_p), mu_error, color=:black, whiskerwidth=6)
scatter!(axm, target, (res_p), color = 1, colormap = :tab10, colorrange = (1, 10), marker=Circle)
# check if residual lies within ylims, if not, draw an arrow to indicate that the point is outside the plot range
for (i, r) in enumerate(res_p)
    if r < ylims_m[1]
        scatter!(axm, [target[i]], [ylims_m[1] + 0.1 * (ylims_m[2] - ylims_m[1])], color = 1, colormap = :tab10, colorrange = (1, 10), marker='↓', markersize=15)
    elseif r > ylims_m[2]
        scatter!(axm, [target[i]], [ylims_m[2] - 0.1 * (ylims_m[2] - ylims_m[1])], color = 1, colormap = :tab10, colorrange = (1, 10), marker='↑', markersize=15)
    end
end

#axislegend(axm, position=legend_position[1],padding=1)

res_w = s2_means .- s2_targets
figs = Figure()
axs = Axis(figs[1,1], title=title, xlabel=xlabel, ylabel=L"(reco - target) $b_w$", yticks=-0.1:0.05:0.25)
ylims!(axs, ylims_s...)
lines!(axs, target, zeros(length(target)), color = 4, colormap = :tab10, colorrange = (1, 10))
errorbars!(axs, target, (res_w), s2_error, color=:black, whiskerwidth=6)
scatter!(axs, target, (res_w),color = 1, colormap = :tab10, colorrange = (1, 10), marker=Circle)
#axislegend(axs, position=legend_position[2], padding=1)
for (i, r) in enumerate(res_w)
    if r < ylims_s[1]
        scatter!(axs, [target[i]], [ylims_s[1] + 0.1 * (ylims_s[2] - ylims_s[1])], color = 1, colormap = :tab10, colorrange = (1, 10), marker='↓', markersize=15)
    elseif r > ylims_s[2]
        scatter!(axs, [target[i]], [ylims_s[2] - 0.1 * (ylims_s[2] - ylims_s[1])], color = 1, colormap = :tab10, colorrange = (1, 10), marker='↑', markersize=15)
    end
end

res_h = h_means .- h_targets
figh = Figure()
axh = Axis(figh[1,1], title=title, xlabel=xlabel, ylabel=L"(reco - target) $b_h$", yticks=-0.1:0.05:0.25)
ylims!(axh, ylims_h...)
lines!(axh, target, zeros(length(target)), color = 4, colormap = :tab10, colorrange = (1, 10))
errorbars!(axh, target, (res_h), h_error, color=:black, whiskerwidth=6)
scatter!(axh, target, (res_h), color = 1, colormap = :tab10, colorrange = (1, 10), marker=Circle)
#axislegend(axh, position=legend_position[2], padding=1)
for (i, r) in enumerate(res_h)
    if r < ylims_h[1]
        scatter!(axh, [target[i]], [ylims_h[1] + 0.1 * (ylims_h[2] - ylims_h[1])], color = 1, colormap = :tab10, colorrange = (1, 10), marker='↓', markersize=15)
    elseif r > ylims_h[2]
        scatter!(axh, [target[i]], [ylims_h[2] - 0.1 * (ylims_h[2] - ylims_h[1])], color = 1, colormap = :tab10, colorrange = (1, 10), marker='↑', markersize=15)
    end
end

fignrmse = Figure()
axnrmse = Axis(fignrmse[1,1], title=title, xlabel=xlabel, ylabel="NRMSE")
ylims!(axnrmse, (0, 0.5)...)
scatter!(axnrmse, target, nrmse_vals, color = 1, colormap = :tab10, colorrange = (1, 10))
for (i, r) in enumerate(nrmse_vals)
    if r < 0
        scatter!(axnrmse, [target[i]], [0.1], color = 1, colormap = :tab10, colorrange = (1, 10), marker='↓', markersize=15)
    elseif r > 0.5
        scatter!(axnrmse, [target[i]], [0.4], color = 1, colormap = :tab10, colorrange = (1, 10), marker='↑', markersize=15)
    end
end

println("Store at $(experiment)")
save(joinpath(experiment, "mu_means_mcse_$(extension)_makie_nolegend.pdf"), figm; pt_per_unit)
save(joinpath(experiment, "s2_means_mcse_$(extension)_makie_nolegend.pdf"), figs; pt_per_unit)
save(joinpath(experiment, "h_means_mcse_$(extension)_makie_nolegend.pdf"), figh; pt_per_unit)
save(joinpath(experiment, "nrmse_$(extension)_makie_nolegend.pdf"), fignrmse; pt_per_unit)