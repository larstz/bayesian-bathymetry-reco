using Pkg
Pkg.activate(".")
Pkg.instantiate()

using CairoMakie
using Serialization
using LaTeXStrings

println(length(ARGS), " arguments provided.")
println("Read in data from disk")

cwd = pwd()
lp_experiment = ARGS[1]

lps = deserialize(joinpath(lp_experiment, "log_posterior_values.jls"))
μs = deserialize(joinpath(lp_experiment, "log_posterior_grid_ms.jls"))
σs = deserialize(joinpath(lp_experiment, "log_posterior_grid_ss.jls"))
p_grid = hcat([[μ, σ] for μ in μs for σ in σs]...)

line_colors = [RGBf(0.1216, 0.4667, 0.7059), RGBf(1.0, 0.498, 0.0549), RGBf(0.1725, 0.6275, 0.1725), RGBf(0.5804, 0.4039, 0.7412), RGBf(0.7373, 0.7412, 0.1333), RGBf(0.0902, 0.7451, 0.8118), RGBf(0.8902, 0.4667, 0.7608), RGBf(0.549, 0.3373, 0.2941), RGBf(0.8392, 0.1529, 0.1569), RGBf(0.498, 0.498, 0.498)]

lp_mat = permutedims(reshape(lps, length(σs), length(μs)))
#replace -Inf with NaN
lp_mat[lp_mat .== -Inf] .= NaN

width_cm = 30
height_cm = 19.5
function cm2px(size)
    return (size[1]*96/ 2.54, size[2]*96/ 2.54)
end

println("Plotting log-posterior")
f = Figure(size=cm2px((width_cm, height_cm)), fontsize=30)
ax = Axis(f[1, 1]; title=latexstring("\\pi_L(H|b)\\pi_\\mathrm{Prior}(b)"), xlabel=latexstring("\\mu"), ylabel=latexstring("\\sigma^2"), xticks=1.5:1.:15.0, yticks=0.0:0.05:0.5, titlesize=32)
cont = contourf!(ax,μs, σs, lp_mat; nan_color=:white, levels=50)
contour!(ax, μs, σs, lp_mat; nan_color=:white,levels=49, linewidth=0.25, color=:white)
Colorbar(f[1, 2], cont)

cd(lp_experiment)

save("log_posterior.png", f)
save("log_posterior.pdf", f)
cd(cwd)
if length(ARGS) == 2
    chain_experiment = ARGS[2]
    files = readdir(chain_experiment)
    chains = filter(x -> occursin(r"chain_[0-9]+.jls", x), files)
    n_chains = length(chains)
    samples = [deserialize(joinpath(chain_experiment, file)) for file in chains]

    println("Plot the MCMC samples")
    for (i,chain) in enumerate(samples)
        μ = chain[:, 1]
        σ = chain[:, 2]
        scatter!(ax, μ[1], σ[1], color=line_colors[i], markersize=12)
        lines!(ax, μ, σ, label="Chain $(i)", color=line_colors[i], linewidth=3)
    end

    println("Plotting max posterior")
    _, idmax = findmax(lps)
    println("Max Posterior: ", idmax)
    max_p = p_grid[:, idmax]
    println("Max Posterior: ", max_p)
    scatter!(ax, max_p[1], max_p[2], color=:black, label="Max Posterior", markersize=12)
    axislegend(ax)

    cd(chain_experiment)

    save("log_posterior_with_chains.png", f, px_per_unit=300/96)
    save("log_posterior_with_chains.pdf", f, px_per_unit=300/96)
end
