using Pkg
Pkg.activate(".")

using BathymetryReco
using DataFrames
using Statistics
using Serialization
using Dates
using LaTeXStrings
using MCMCChains
using Plots

include("my_theme.jl")
theme(:custom)

# Half size with no whitespace
half_width = textwidth / 2
half_height = plot_height/1.5
plot_size = (half_width, half_height)

experiment = ARGS[1]
chain_list = String[]
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
    xlabel = "target " * latexstring("b_w")
    mutarget = "target "*latexstring("b_p=4.0")
    s2target = "target "*latexstring("b_w")
    extension = "width"
    if include_title
        title = "Target "*latexstring("b_w")*" vs reconstructed "*latexstring("b_w")
    end
    legend_position = [:bottomright, :topleft]
else
    target = mu_targets
    xlabel = "target " * latexstring("b_p")
    mutarget = "target "*latexstring("b_p")
    s2target = "target "*latexstring("b_w=0.05")
    extension = "position"
    if include_title
        title = "Target "*latexstring("b_p")*" vs reconstructed "*latexstring("b_p")
    end
    legend_position = [:topleft, :bottomleft]
end

pm = plot(target, zeros(length(mu_targets)), label=mutarget, color=:brown4, linewidth=0.75,
title=title, xlabel=xlabel, ylabel="reco - target " * latexstring("b_p"),
size=plot_size, margin=0Plots.mm, ylims=(-1, 1))
plot!(target, mu_means.-mu_targets, seriestype=:scatter, yerror=mu_error, label="reco " * latexstring("b_p"), color=palette(:default)[1])
plot!(pm, legend_position=:bottomleft)

ps = plot(target, zeros(length(s2_targets)), label=s2target, color=:brown4, linewidth=0.75,
 title=title, xlabel=xlabel, ylabel="reco - target " * latexstring("b_w"), size=plot_size, margin=0Plots.mm, ylims=(-0.1, 0.1))
scatter!(target, s2_means.-s2_targets, yerror=s2_error, label="reco " * latexstring("b_w"), color=palette(:default)[1])
plot!(ps, legend_position=:bottomleft)

savefig(pm, joinpath(experiment, "mu_means_mcse_tex_$(extension).pdf"))
savefig(ps, joinpath(experiment, "s2_means_mcse_tex_$(extension).pdf"))