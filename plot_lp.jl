using Pkg
Pkg.activate(".")
Pkg.instantiate()

using CairoMakie
using Serialization
using LaTeXStrings
using Distributions

pt_per_unit = 1
figsize = (390, round(Int, 390*0.618))
set_theme!(theme_latexfonts(),
            fontsize=12,
            size=figsize,
            figure_padding=0)

function get_prior(prior, points)
    return sum(logpdf.(prior, points), dims=1)
end

println("Read in data from disk")

cwd = pwd()
lp_experiment = ARGS[1]

test_prior = false

lps = getindex.(deserialize(joinpath(lp_experiment, "log_posterior_values.jls")),1)
lls = getindex.(deserialize(joinpath(lp_experiment, "log_posterior_values.jls")),2)
prs = getindex.(deserialize(joinpath(lp_experiment, "log_posterior_values.jls")),3)
μs = deserialize(joinpath(lp_experiment, "log_posterior_grid_ms.jls"))
σs = deserialize(joinpath(lp_experiment, "log_posterior_grid_ss.jls"))
p_grid = hcat([[μ, σ] for μ in μs for σ in σs]...)

line_colors = [RGBf(0.1216, 0.4667, 0.7059), RGBf(1.0, 0.498, 0.0549), RGBf(0.1725, 0.6275, 0.1725), RGBf(0.5804, 0.4039, 0.7412), RGBf(0.7373, 0.7412, 0.1333), RGBf(0.0902, 0.7451, 0.8118), RGBf(0.8902, 0.4667, 0.7608), RGBf(0.549, 0.3373, 0.2941), RGBf(0.8392, 0.1529, 0.1569), RGBf(0.498, 0.498, 0.498)]

lp_mat = permutedims(reshape(lps, length(σs), length(μs)))./1000
#replace -Inf with NaN
lp_mat[lp_mat .== -Inf] .= NaN
ll_mat = permutedims(reshape(lls, length(σs), length(μs)))
pr_mat = permutedims(reshape(prs, length(σs), length(μs)))


println("Plotting log-posterior")
f = Figure()
ax = Axis(f[1, 1]; title=latexstring("\\log\\pi_L(H|b)+\\log\\pi_\\mathrm{Prior}(b)"), xlabel=latexstring("b_p"), ylabel=latexstring("b_w"), xticks=1.5:1.:15.0, yticks=0.0:0.05:0.5)
cont = contourf!(ax,μs, σs, lp_mat; nan_color=:white, levels=50)
contour!(ax, μs, σs, lp_mat; nan_color=:white,levels=49, linewidth=0.25, color=:white)
Colorbar(f[1, 2], cont)
Label(f[1, 2, Top()], halign=:left, text=L"\times10^3")

println("Plotting log-likelihood")
fll = Figure()
axll = Axis(fll[1, 1]; title=latexstring("\\log\\pi_L(H|b)"), xlabel=latexstring("b_p"), ylabel=latexstring("b_w"), xticks=1.5:1.:15.0, yticks=0.0:0.05:0.5)
contll = contourf!(axll,μs, σs, ll_mat; nan_color=:white, levels=50)
contour!(axll, μs, σs, ll_mat; nan_color=:white,levels=49, linewidth=0.25, color=:white)
Colorbar(fll[1, 2], contll)

if test_prior
    test_prior = [Normal(4.0, 1.0), Uniform(0.0, 1.0)]
    prior_vals = get_prior(test_prior, p_grid)
    pr_mat_test = permutedims(reshape(prior_vals, length(σs), length(μs)))

    fpr = Figure()
    axpr = Axis(fpr[1, 1]; title=latexstring("\\log\\pi_L(H|b)+\\log\\pi_\\mathrm{Prior}(b)"), xlabel=latexstring("b_p"), ylabel=latexstring("b_w"), xticks=1.5:1.:15.0, yticks=0.0:0.05:0.5)
    contpr = contourf!(axpr,μs, σs, pr_mat_test.+ll_mat; nan_color=:white, levels=50)
    contour!(axpr, μs, σs, pr_mat_test.+ll_mat; nan_color=:white,levels=49, linewidth=0.25, color=:white)
    Colorbar(fpr[1, 2], contpr)
end

cd(lp_experiment)

save("log_posterior.png", f)
save("log_posterior.pdf", f; pt_per_unit)

save("log_likelihood.png", fll)
save("log_likelihood.pdf", fll)

if test_prior
    save("log_prior_N401_N005001_plus_likelihood.png", fpr)
    save("log_prior_N401_N005001_plus_likelihood.pdf", fpr)
end

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
        scatter!(axpr, μ[1], σ[1], color=line_colors[i], markersize=5)
        lines!(axpr, μ, σ, label="Chain $(i)", color=line_colors[i], linewidth=2)
    end

    println("Plotting max posterior")
    _, idmax = findmax(lps)
    println("Max Posterior: ", idmax)
    max_p = p_grid[:, idmax]
    println("Max Posterior: ", max_p)
    scatter!(axpr, max_p[1], max_p[2], color=:black, label="Max Posterior", markersize=5)
    axislegend(axpr)

    cd(chain_experiment)

    save("log_posterior_with_chains.png", fpr, px_per_unit=300/96)
    save("log_posterior_with_chains.pdf", fpr, px_per_unit=300/96)
end
