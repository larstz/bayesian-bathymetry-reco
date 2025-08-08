using Pkg
Pkg.activate(".")
using Serialization
using Plots
using Statistics
using BathymetryReco
using MCMCChains

println("#############################\nRead in chain" )

exp = "data/results/waterchannel_exact_bathy_2025-08-07-15-50-40/"
chain = deserialize(joinpath(exp, "chain_1.jls"))
obs,_ = load_observation("data/toy_measurement/waterchannel_exact_bathy/")

bathy = chain[:,1:64]
lp = chain[:,65]
ar = chain[:,66]

burnin = 2000

xs = range(1.5,15.0,64)
exact_b = exp_bathymetry(xs)

println("#############################\nCreate Gif" )

# anim = @animate for (i, b) in enumerate(eachrow(bathy))
#     plot(xs, b, label="chain $i", ylims=(-0.01,0.21), xlabel="x", ylabel="b(x)",title="Bathymetry Sample progression")
#     plot!(xs, exact_b, label="True Bathymetry", color=:black)
# end every 10

# gif(anim, exp*"/chain_progression.gif", fps=10)

println("#############################\nCreate error plot" )

mean_bathy = vec(mean(bathy[burnin:end,:], dims=1))
bathy_nrmse = sqrt.(mean((mean_bathy .- exact_b).^2)) ./ (maximum(exact_b) .- minimum(exact_b))*100
bathy_l2 = sqrt(sum((mean_bathy .- exact_b).^2))/sqrt(sum(exact_b.^2))*100
bathy_linf = maximum(abs.(mean_bathy .- exact_b))/maximum(exact_b)*100
mcmc_chain = Chains(bathy)
grid_error = mcse(mcmc_chain)[:, :mcse]

error_plot = scatter(xs, mean_bathy, yerror=grid_error, label="Mean of last $(length(mean_bathy)) samples", markersize=2,
     ylims=(-0.01,0.21), xlabel="x", ylabel="b(x)", title="Bathymetry Sample Mean: NRMSE = $(round(bathy_nrmse, digits=3))%")
plot!(error_plot, xs, mean_bathy; label="NRMSE = $(round(bathy_nrmse, digits=3))% \n l2 = $(round(bathy_l2, digits=3))% \n linf = $(round(bathy_linf, digits=3))%")
plot!(error_plot, xs, exact_b, label="True Bathymetry", color=:black)
savefig(error_plot, exp*"/mean_bathy_errorbars.png")

mp = plot(xs, mean_bathy, label="Mean of last 500 samples",
     ylims=(-0.01,0.21), xlabel="x", ylabel="b(x)", title="Bathymetry Sample Mean: NRMSE = $(round(bathy_nrmse, digits=3))%")
plot!(mp, xs, exact_b, label="True Bathymetry", color=:black)
savefig(mp, exp*"/mean_bathy_with_error.png")