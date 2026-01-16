export Posterior, logprior, loglikelihood
struct Posterior{T1<:Distribution, T2<:Distribution}
    prior::Union{T1, Array{T1}}
    likelihood::Union{T2, Array{T2}}
end

function logprior(p::Posterior, θ)
    logp = 0.0
    if length(θ) <= 4 # üarametrized bathymetry
        return sum(logpdf.(p.prior, θ))
    end
    for lp in p.prior
        logp += sum(logpdf(lp, θ))
    end
    return logp
end

function loglikelihood(p::Posterior, θ)
    return logpdf(p.likelihood, θ')
end

function loglikelihood(p::Posterior, θ, obs)
    return -length(θ)*0.5 * log(2π) -length(θ)*log(p.likelihood.σ) - 1/ (2 * p.likelihood.σ^2)* sum((θ - obs).^2 )
end

export Proposal, RandomWalkProposal, pCNProposal, proposal
abstract type Proposal end

struct RandomWalkProposal<:Proposal
    γ::Union{Float64, Array{Float64,1}}
    C::PDMat
end

# Constructor for when you only have gamma and dimension
RandomWalkProposal(γ::Union{Float64, Array{Float64,1}}, dim::Int) = RandomWalkProposal(γ, PDMat(Matrix(I, dim, dim)))
struct pCNProposal<:Proposal
    β::Float64
    C::PDMat
end

function proposal(θ, proposal::RandomWalkProposal)
    return rand(MvNormal(θ, proposal.γ.^2 .* proposal.C))
end

function proposal(θ, proposal::pCNProposal)
    return rand(MvNormal(√(1-proposal.β^2) .* θ,proposal.β^2 .*proposal.C))
end

function proposal(θ, proposal::Distribution)
    return θ .+ rand(proposal)
end

export SqExpMvNormal
struct SqExpMvNormal
    dim::Int
    lengthscale::Int64 # number of grid points
    variance::Float64
end

function Base.show(io::IO, sp::SqExpMvNormal)
    print(io, typeof(sp), "(")
    print(io, "dim: ", sp.dim, ", ")
    print(io, "lengthscale: ", sp.lengthscale, ", ")
    print(io, "variance: ", sp.variance)
    print(io, ")")
end

# Convert to MvNormal
function Distributions.MvNormal(p::SqExpMvNormal)
    xs = LinRange(1, p.dim, p.dim)
    C = p.variance .* exp.(-((xs .- xs').^2) / (2*p.lengthscale^2))
    return MvNormal(zeros(p.dim), PDMat(Matrix(C)))
end

export mcmc_model
struct mcmc_model
    posterior::Posterior
    forward::Function
    observation::observation_data
    proposal::Union{Distribution, Proposal}
end

export logjoint
function logjoint(model::mcmc_model, θ)
    log_prior = sum(logprior(model.posterior, θ))
    if log_prior == -Inf
        return -Inf, 0.0, -Inf
    end

    try
        sim_observations = model.forward(θ)
        log_likelihood = sum(loglikelihood(model.posterior, sim_observations .- model.observation.H))
        return log_prior + log_likelihood, log_likelihood, log_prior
    catch err
        if isa(err, DimensionMismatch) || isa(err, BoundsError)
            println("Dimension mismatch or bounds error in forward model / likelihood.")
            println("Current parameters: ", θ)
            return -Inf, -Inf, log_prior
        else
            rethrow(err)
        end
    end
end

export sample_chain
function sample_chain(model::mcmc_model, n, initial_θ; verbose=false, logging=Progress(n), γ=0.1, burn_in=0)
    chain = zeros(n-burn_in+1, length(initial_θ)+4)
    θ = initial_θ
    lpost, ll, lp = logjoint(model, θ)
    chain[1, :] = [θ..., lpost, ll, lp, 1.0]
    acceptance_rate = 1.0
    accepted = 1
    for i in 1:n

        θ_new = proposal(θ, model.proposal)
        lpost_new, ll_new, lp_new = logjoint(model, θ_new)

        if (rand()) < exp(lpost_new - lpost)
            accepted += 1
            θ = θ_new
            lpost = lpost_new
            ll = ll_new
            lp = lp_new
        end
        acceptance_rate = accepted / (i+1)
        if i > burn_in
            chain[i-burn_in+1, :] = [θ..., lpost, ll, lp, acceptance_rate]
        end
        if verbose
            next!(logging, showvalues = [("iteration count",i)])
        end
    end
    return chain
end

function sample_chain(model::mcmc_model, setup::mcmc_setup; kargs...)
    return sample_chain(model, setup.n, setup.init;γ=setup.γ, burn_in=setup.burn_in, kargs...)
end

function sample_chain(model::mcmc_model, setup::mcmc_setup, initial_θ; kargs...)
    return sample_chain(model, setup.n, initial_θ;γ=setup.γ, burn_in=setup.burn_in, kargs...)
end
