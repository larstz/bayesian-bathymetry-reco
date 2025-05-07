using CairoMakie
using Serialization

println("Read in data from disk")
lp_experiment = ARGS[1]

lps = deserialize(joinpath(lp_experiment, "log_posterior_values.jls"))
μs = deserialize(joinpath(lp_experiment, "log_posterior_grid_ms.jls"))
σs = deserialize(joinpath(lp_experiment, "log_posterior_grid_ss.jls"))
p_grid = hcat([[μ, σ] for μ in μs for σ in σs]...)

lp_mat = permutedims(reshape(lps, length(σs), length(μs)))
#replace -Inf with NaN
lp_mat[lp_mat .== -Inf] .= NaN

println("Plotting log-posterior")
f = Figure()
ax = Axis(f[1, 1]; title="Log Posterior", xlabel="μ", ylabel="σ", xticks=1.5:1.:15.0, yticks=0.0:0.1:2.0)
cont = contourf!(ax,μs, σs, lp_mat; nan_color=:white, levels=50)
contour!(ax, μs, σs, lp_mat; nan_color=:white,levels=49, linewidth=0.25, color=:white)
Colorbar(f[1, 2], cont)

if length(ARGS) > 2
    chain_experiment = ARGS[2]
    files = readdir(chain_experiment)
    chains = filter(x -> occursin(r"chain_[0-9]+.jls", x), files)
    n_chains = length(chains)
    samples = [deserialize(joinpath(chain_experiment, file)) for file in chains]

    println("Plot the MCMC samples")
    for (i,chain) in enumerate(samples)
        μ = chain[:, 1]
        σ = chain[:, 2]
        scatter!(ax, μ[1], σ[1])
        lines!(ax, μ, σ, label="Chain $(i)")
    end

    println("Plotting max posterior")
    _, idmax = findmax(lps)
    println("Max Posterior: ", idmax)
    max_p = p_grid[:, idmax]
    println("Max Posterior: ", max_p)
    scatter!(ax, max_p[1], max_p[2], color=:black, label="Max Posterior")
    axislegend(ax)
end


cd(lp_experiment)

save("log_posterior.png", f)
save("log_posterior.pdf", f)