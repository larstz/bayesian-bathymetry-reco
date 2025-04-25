using CairoMakie
using Serialization

println("Read in data from disk")
#lp_experiment = ARGS[1]
chain_experiment = "./data/results/waterchannel_exact_bathy_2025-04-15-16-08-57" #ARGS[2]
lps = deserialize("log_posterior_values.jls")
#grid_p = deserialize("log_posterior_grid.jls")
μs = LinRange(1.5,15.0,100) # deserialize("log_posterior_ms.jls")
σs = LinRange(0.0, 2.0, 50) # deserialize("log_posterior_ss.jls")
p_grid = hcat([[μ, σ] for μ in μs for σ in σs]...)'

lp_mat = permutedims(reshape(lps, length(σs), length(μs)))
#replace -Inf with NaN
lp_mat[lp_mat .== -Inf] .= NaN

println("Plotting log-posterior")
f = Figure()
ax = Axis(f[1, 1]; title="Log Posterior", xlabel="μ", ylabel="σ", xticks=1.5:1.:15.0, yticks=0.0:0.1:2.0, palette=(color = [:red,:orange, :brown],))
contourf!(ax,μs, σs, lp_mat; nan_color=:white, colorbar=true, levels=50)
contour!(ax, μs, σs, lp_mat; nan_color=:white,levels=49, linewidth=0.25, color=:white)

files = readdir(chain_experiment)
chains = filter(x -> occursin(r"chain_[0-9]+.jls", x), files)
n_chains = length(chains)
samples = [deserialize(joinpath(experiment, file)) for file in chains]

println("Plot the MCMC samples")
for (i,chain) in enumerate(samples)
    μ = chain[:, 1]
    σ = chain[:, 2]
    scatter!(ax, μ[1], σ[1])
    lines!(ax, μ, σ, alpha=0.5, label="Chain $(i)")
end

println("Plotting max posterior")
_, idmax = findmax(lps)
max_p = p_grid[:, idmax]
scatter!(ax, max_p[1], max_p[2], color=:black, label="Max Posterior")
axislegend(ax)

save("log_posterior.png", f)
save("log_posterior.pdf", f)