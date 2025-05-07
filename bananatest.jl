using BathymetryReco
using Distributions

struct HybridRosenbrock{Ta,Tb}<: ContinuousUnivariateDistribution
    a::Ta
    b::Tb
end
struct Dummy{T}<: ContinuousUnivariateDistribution
    a::T
end

function Distributions.logpdf(D::HybridRosenbrock, x::AbstractArray{<:Real})
    (-1/10.0 * x[1]^2  - 1/10.0 * x[2]^2  - 2.0*(x[2] - x[1]^2)^2)#*10^-5
    #logpdf(MvNormal([0.0, 0.0], .5), x) +
    #logpdf(MvNormal([2.0, 2.0], .5), x)
end

function Distributions.logpdf(D::Dummy, x::AbstractArray{<:Real})
    0.0
end

function BathymetryReco.logprior(p::Posterior, x)
    return logpdf(p.prior, x)
end

function BathymetryReco.loglikelihood(p::Posterior, x)
    return logpdf(p.likelihood, x)
end

forward(x) = x
ll = HybridRosenbrock(1.0, 1.0)
obs = [0.0, 0.0]
obs_data = observation_data([0.0], [0.0], [0.0], obs, 0.0, [0.0])
prior = Dummy(1.0)
posterior = Posterior(prior, ll)

model = mcmc_model(posterior, forward, obs_data)
n = 1000000
initial_θ = [2.0, 4.0]
chain = sample_chain(model, n, initial_θ; γ=10.0, burn_in=0)

using Plots
println("Chain statistics:")
println(maximum(chain[:, 3]))
println(minimum(chain[:, 3]))
p = plot(chain[:, 1], chain[:, 2], title="Hybrid Rosenbrock", xlabel="x1", ylabel="x2", linez=range(0.0, stop=1.0, length=length(chain[:,1])))
savefig("bananatest/banana10_scaled.png")
h = histogram2d(chain[:, 1], chain[:, 2], bins=(30,50), title="Hybrid Rosenbrock", xlabel="x1", ylabel="x2")
savefig("bananatest/banana_hist10_scaled.png")
lp = plot(chain[:, 3], title="Hybrid Rosenbrock", xlabel="iteration", ylabel="lp")
savefig("bananatest/banana_lp10_scaled.png")