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

experiment = ARGS[1]
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

n_targets = length(targets_vec)
n_chains = round(Int64, length(chains)/n_targets)

samples = deserialize.(chains)

# adapt to new format also storing loglikelihood and logprior
for (i, sample) in enumerate(samples)
    if size(sample,2) >= 6
        samples[i] = sample[:, [1,2,4,end]] # keep only mu, s2, logp, ar
    end
end

chain_tensor = reshape(hcat(samples...), size(samples[1])[1], size(samples[1])[2], n_chains, n_targets)
chain_tensor = chain_tensor[:, 1:4, :, :]
burn_in = 300
burned_tensor = chain_tensor[burn_in:end, :, :, :]
param_names = ["mu", "s2", "lp", "ar"]
mean_lp = mean(burned_tensor[:, 3, :, :], dims=1)
d_lp = maximum(mean_lp, dims=2) .- mean_lp

exp_id = [reshape(d_lp[:,:,i].<2, 7) for i in 1:n_targets]

mu_means = zeros(n_targets)
s2_means = zeros(n_targets)
mu_error = zeros(n_targets)
s2_error = zeros(n_targets)
ci_low = zeros(n_targets, 2)
ci_high = zeros(n_targets, 2)
for (i, id) in enumerate(exp_id)
    mu_means[i] = mean(chain_tensor[:, 1, id, i])
    s2_means[i] = mean(chain_tensor[:, 2, id, i])
    temp_chain = Chains(chain_tensor[:, :, id, i], param_names, Dict(:internal => [:lp, :ar]))
    error = mcse(temp_chain)
    mu_error[i] = error[:mu,:mcse]
    s2_error[i] = error[:s2,:mcse]
    ci_low[i, :] = hpd(temp_chain)[:, :lower]
    ci_high[i,:] = hpd(temp_chain)[:, :upper]
end

title = ""
include_title = false
if occursin("width_test", experiment)
    target = s2_targets
    xlabel = L"target $b_w$"
    mutarget = L"target $b_p=4.0$"
    s2target = L"target $b_w$"
    extension = "width"
    if include_title
        title = L"Target $b_w$ vs reconstructed $b_w$"
    end
    legend_position = [:rb, :lt]
    ylims_m = (-1, 1)
    ylims_s = (-0.05, 0.25)
else
    target = mu_targets
    xlabel = L"target $b_p$"
    mutarget = L"target $b_p$"
    s2target = L"target $b_w=0.05$"
    extension = "position"
    if include_title
        title = L"Target $b_p$ vs reconstructed $b_p$"
    end
    legend_position = [:rb, :lt]
    ylims_m = (-1, 1)
    ylims_s = (-0.05, 0.25)
end

figm = Figure()
axm = Axis(figm[1,1], title=title, xlabel=xlabel, ylabel=L"(reco-target) $b_p$", yticks=-1:0.5:1)
ylims!(axm, ylims_m...)

lines!(axm, target, zeros(length(target)), color = 4, colormap = :tab10, colorrange = (1, 10))
errorbars!(axm, target, (mu_means.-mu_targets), mu_error, color=:black, whiskerwidth=6)
scatter!(axm, target, (mu_means.-mu_targets), color = 1, colormap = :tab10, colorrange = (1, 10), marker=Circle)
#axislegend(axm, position=legend_position[1],padding=1)

figs = Figure()
axs = Axis(figs[1,1], title=title, xlabel=xlabel, ylabel=L"(reco - target) $b_w$", yticks=-0.1:0.05:0.25)
ylims!(axs, ylims_s...)
lines!(axs, target, zeros(length(target)), color = 4, colormap = :tab10, colorrange = (1, 10))
errorbars!(axs, target, (s2_means.-s2_targets), s2_error, color=:black, whiskerwidth=6)
scatter!(axs, target, (s2_means.-s2_targets),color = 1, colormap = :tab10, colorrange = (1, 10), marker=Circle)
#axislegend(axs, position=legend_position[2], padding=1)

save(joinpath(experiment, "mu_means_mcse_$(extension)_makie_nolegend.pdf"), figm; pt_per_unit)
save(joinpath(experiment, "s2_means_mcse_$(extension)_makie_nolegend.pdf"), figs; pt_per_unit)