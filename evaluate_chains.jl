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
chain_tensor = reshape(hcat(samples...), size(samples[1])[1], size(samples[1])[2], n_chains, n_targets)
burn_in = 300
burned_tensor = chain_tensor[burn_in:end, :, :, :]
param_names = ["mu", "s2", "ll", "lp", "lprior", "ar"]
mean_lp = mean(burned_tensor[:, 4, :, :], dims=1)
d_lp = maximum(mean_lp, dims=2) .- mean_lp

exp_id = [reshape(d_lp[:,:,i].<10, 7) for i in 1:n_targets]

mu_means = zeros(n_targets)
s2_means = zeros(n_targets)
mu_error = zeros(n_targets)
s2_error = zeros(n_targets)
for (i, id) in enumerate(exp_id)
    mu_means[i] = mean(chain_tensor[:, 1, id, i])
    s2_means[i] = mean(chain_tensor[:, 2, id, i])
    temp_chain = Chains(chain_tensor[:, :, id, i], param_names, Dict(:internal => [:lp, :ll, :lprior, :ar]))
    error = mcse(temp_chain)
    mu_error[i] = error[:mu,:mcse]
    s2_error[i] = error[:s2,:mcse]
end

if occursin("width_test", experiment)
    target = s2_targets
    xlabel = "s2 target"
    mutarget = "Target=4.0"
    s2target = "Target=s2"
else
    target = mu_targets
    xlabel = "mu target"
    mutarget = "Target=mu"
    s2target = "Target=0.05"
end

pm = scatter(target, mu_means, yerror=mu_error, label="Mean of mu", xlabel=xlabel, title="Mean of mu across plausible chains")
plot!(pm, target, mu_targets, label=mutarget, linestyle=:dash, color=:red)
ps = scatter(target, s2_means, yerror=s2_error, label="Mean of s2", xlabel=xlabel, title="Mean of s2 across plausible chains")
plot!(ps, target, s2_targets, label=s2target, linestyle=:dash, color=:red)
savefig(pm, joinpath(experiment, "mu_means.png"))
savefig(ps, joinpath(experiment, "s2_means.png"))