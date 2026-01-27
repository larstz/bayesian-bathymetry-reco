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

experiment = ARGS[1]
chain_list = String[]
for (exp,_, files) in walkdir(experiment)
    if splitdir(exp)[1]*"/" == experiment
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

exp_id = [reshape(d_lp[:,:,i].<10, 7) for i in 1:n_targets]

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

if occursin("width_test", experiment)
    target = s2_targets
    xlabel = L"b^\dagger_w"
    mutarget = L"b^\dagger_p=4.0"
    s2target = L"Target=b^\dagger_w"
else
    target = mu_targets
    xlabel = L"b^\dagger_p"
    mutarget = L"b^\dagger_p"
    s2target = L"b^\dagger_w=0.05"
end

pm = scatter(target, mu_means, yerror=(mu_means.-ci_low[:,1], ci_high[:,1].-mu_means),
title=L"\textrm{True} \ b^\dagger_p  \ \textrm{vs reconstructed} \ \hat{b}_p", xlabel=xlabel, ylabel=L"\hat{b_p}", label=L"\hat{b}_p")
plot!(pm, target, mu_targets, label=mutarget, linestyle=:dash, color=:red)
ps = scatter(target, s2_means, yerror=(s2_means.-ci_low[:,2], ci_high[:,2].-s2_means),
 title=L"\textrm{True} \ b^\dagger_w \ \textrm{vs reconstructed} \ \hat{b}_w", xlabel=xlabel, ylabel=L"\hat{b}_w", label=L"\hat{b}_w")
plot!(ps, target, s2_targets, label=s2target, linestyle=:dash, color=:red)
savefig(pm, joinpath(experiment, "mu_means_ci_tex.pdf"))
savefig(ps, joinpath(experiment, "s2_means_ci_tex.pdf"))