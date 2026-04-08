using Pkg
Pkg.activate(".")
Pkg.instantiate()

using CairoMakie
using Serialization
using LaTeXStrings
using Distributions
using BathymetryReco

pt_per_unit = 1
figsize = (390, round(Int, 390*0.618))
set_theme!(theme_latexfonts(),
            fontsize=12,
            size=figsize,
            figure_padding=2)

function get_prior(prior, points)
    return sum(logpdf.(prior, points), dims=1)
end

function finite_minmax(mat)
    finite_vals = mat[isfinite.(mat)]
    return minimum(finite_vals), maximum(finite_vals)
end

function common_colorrange(mats...)
    mins = Float64[]
    maxs = Float64[]
    for mat in mats
        mat_min, mat_max = finite_minmax(mat[:, 1:50])
        println("Matrix min: ", mat_min, " max: ", mat_max)
        push!(mins, mat_min)
        push!(maxs, mat_max)
    end
    return (maximum(mins), maximum(maxs))
end

println("Read in data from disk")

cwd = pwd()
lp_experiment = "data/results/lp_scan/lp_scan_2026-04-07-09-43-57/"#ARGS[1]

test_prior = true


scale_vals = 1000
xlabel = latexstring("b_p")
ylabel = latexstring("b_w")
xlims = (1.5, 12.5)
ylims = (0.01, 0.5)

lps = getindex.(deserialize(joinpath(lp_experiment, "log_posterior_values.jls")),1)
lls = getindex.(deserialize(joinpath(lp_experiment, "log_posterior_values.jls")),2)
prs = getindex.(deserialize(joinpath(lp_experiment, "log_posterior_values.jls")),3)
μs = deserialize(joinpath(lp_experiment, "log_posterior_grid_ms.jls"))
σs = deserialize(joinpath(lp_experiment, "log_posterior_grid_ss.jls"))
p_grid = hcat([[μ, σ] for μ in μs for σ in σs]...)

lp_mat = permutedims(reshape(lps, length(σs), length(μs)))./scale_vals
#replace -Inf with NaN
lp_mat[lp_mat .== -Inf] .= NaN
ll_mat = permutedims(reshape(lls, length(σs), length(μs)))./scale_vals

priors = [[Uniform(1.5, 12.5), Uniform(0.0, 1.0)],
          [Normal(4.0, 0.1), Uniform(0.0, 1.0)],
          [Normal(4.0, 0.01), Uniform(0.0, 1.0)]]

lp_test_mats = []
if test_prior
    for prior in priors
        prior_vals = get_prior(prior, p_grid)
        pr_mat_test = permutedims(reshape(prior_vals, length(σs), length(μs)))
        lp_test_mat = pr_mat_test./scale_vals .+ ll_mat
        push!(lp_test_mats, lp_test_mat)
    end
end

max_lps = [findmax(lpt) for lpt in lp_test_mats]

common_cr = test_prior ? common_colorrange(lp_mat, ll_mat, lp_test_mats...) : common_colorrange(lp_mat, ll_mat)
levels = range(common_cr[1], common_cr[2], length=50)

println("Plotting log-posterior")
f = Figure()
ax = Axis(f[1, 1]; title=latexstring("\\log\\pi_L(H|b)+\\log\\pi_\\mathrm{Prior}(b)"), xlabel=xlabel, ylabel=ylabel, limits=(xlims, ylims), xticks=1.5:1.:12.5, yticks=[0.01,0.05:0.05:0.5...])
cont = contourf!(ax,μs, σs, lp_mat; nan_color=:white, levels=levels)
Colorbar(f[1, 2], cont)
Label(f[1, 2, Top()], halign=:left, text=L"\times10^3")

println("Plotting log-likelihood")
fll = Figure()
axll = Axis(fll[1, 1]; title=latexstring("\\log\\pi_L(H|b)"), xlabel=xlabel, ylabel=ylabel, limits=(xlims, ylims),  xticks=1.5:1.:12.5, yticks=[0.01,0.05:0.05:0.5...])
contll = contourf!(axll,μs, σs, ll_mat; nan_color=:white, levels=levels)
Colorbar(fll[1, 2], contll)
Label(fll[1, 2, Top()], halign=:left, text=L"\times10^3")

test_plots = []
if test_prior
    for lp_test in lp_test_mats
        println("Plotting test log-posterior with prior")
        fpr = Figure()
        axpr = Axis(fpr[1, 1]; title=latexstring("\\log\\pi_L(H|b)+\\log\\pi_\\mathrm{Prior}(b)"), xlabel=xlabel, ylabel=ylabel, limits=(xlims, ylims),xticks=1.5:1.:12.5, yticks=[0.01,0.05:0.05:0.5...])
        contpr = contourf!(axpr,μs, σs, lp_test; nan_color=:white, levels=levels, extendlow=:auto)
        Colorbar(fpr[1, 2], contpr)
        Label(fpr[1, 2, Top()], halign=:left, text=L"\times10^3")
        push!(test_plots, [fpr])
    end
end
cd(lp_experiment)

save("log_posterior.png", f)
save("log_posterior.pdf", f; pt_per_unit)

save("log_likelihood.png", fll)
save("log_likelihood.pdf", fll; pt_per_unit)

if test_prior
    for (i, fpr) in enumerate(test_plots)
        save("log_prior_$(priors[i])_plus_likelihood.png", fpr[1])
        save("log_prior_$(priors[i])_plus_likelihood.pdf", fpr[1]; pt_per_unit)
    end
end

cd(cwd)
if length(ARGS) == 2
    line_colors = [RGBf(0.1216, 0.4667, 0.7059), RGBf(1.0, 0.498, 0.0549), RGBf(0.1725, 0.6275, 0.1725), RGBf(0.5804, 0.4039, 0.7412), RGBf(0.7373, 0.7412, 0.1333), RGBf(0.0902, 0.7451, 0.8118), RGBf(0.8902, 0.4667, 0.7608), RGBf(0.549, 0.3373, 0.2941), RGBf(0.8392, 0.1529, 0.1569), RGBf(0.498, 0.498, 0.498)]

    chain_experiment = ARGS[2]
    files = readdir(chain_experiment)
    chains = filter(x -> occursin(r"chain_[0-9]+.jls", x), files)
    n_chains = length(chains)
    samples = [deserialize(joinpath(chain_experiment, file)) for file in chains]

    config = load_config(joinpath(chain_experiment, "experiment_config.toml"))
    prior_settings = config.mcmc_params.prior

    prior_dist = Vector{Distribution}()
    for (i, prior_type) in enumerate(prior_settings.type)
        prior_param = [prior_settings.loc[i], prior_settings.scale[i]]
        if prior_type == "normal"
            push!(prior_dist, Normal(prior_param...))
        elseif prior_type == "uniform"
            push!(prior_dist, Uniform(prior_param...))
        else
            error("Unsupported prior type: $prior_type")
        end
    end

    plot_id = findfirst(x->x == prior_dist, priors)
    f_chn = test_plots[plot_id][1]
    println("Plot the MCMC samples")
    for (i,chain) in enumerate(samples)
        μ = chain[:, 1]
        σ = chain[:, 2]
        scatter!(f_chn.content[1], μ[1], σ[1], color=line_colors[i], markersize=5)
        lines!(f_chn.content[1], μ, σ, color=line_colors[i], linewidth=2)
    end

    println("Plotting max posterior")
    _, idmax = max_lps[plot_id]
    println("Max Posterior: ", idmax)
    max_p = [μs[idmax[1]], σs[idmax[2]]]
    println("Max Posterior: ", max_p)
    scatter!(f_chn.content[1], max_p[1], max_p[2], color=:black, markersize=5)
    chain_elem = [LineElement(color=line_colors[i], linestyle=nothing,
        linepoints=Point2f[(0, i/(n_chains+1)), (1, i/(n_chains+1))]) for i in 1:n_chains]
    max_elem = MarkerElement(color=:black, marker=:circle, markersize=5)
    axislegend(f_chn.content[1], [chain_elem, max_elem], ["Chains", "Max Posterior"])

    cd(chain_experiment)

    save("log_posterior_with_chains.png", f_chn)
    save("log_posterior_with_chains.pdf", f_chn; pt_per_unit)
end
