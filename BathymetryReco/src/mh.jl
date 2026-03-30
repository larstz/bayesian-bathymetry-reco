export Posterior, logprior, loglikelihood
"""
    Posterior(prior, likelihood)

The `Posterior` struct encapsulates the prior and likelihood distributions for a
Bayesian inference problem.

# Arguments

  - `prior`: A distribution or an array of distributions representing the prior
             belief about the parameters.
  - `likelihood`: A distribution or an array of distributions representing the likelihood
                  of the observed data
"""
struct Posterior{T1<:Distribution,T2<:Distribution}
    prior::Union{T1,Array{T1}}
    likelihood::Union{T2,Array{T2}}
end

"""
    logprior(p::Posterior, θ)

Compute the log prior probability of parameters `θ` given the `Posterior` distribution `p`.
"""
function logprior(p::Posterior, θ)
    logp = 0.0
    if length(θ) <= 4 # parametrized bathymetry
        return sum(logpdf.(p.prior, θ))
    end
    for lp in p.prior
        logp += sum(logpdf(lp, θ))
    end
    return logp
end

"""
    loglikelihood(p::Posterior, θ)

Compute the log likelihood of parameters `θ` given the `Posterior` distribution `p`.
"""
function loglikelihood(p::Posterior, θ)
    return logpdf(p.likelihood, θ')
end

export Proposal, RandomWalkProposal, pCNProposal, proposal
abstract type Proposal end

"""
    RandomWalkProposal(γ, C)

A random walk proposal distribution for MCMC sampling.

# Arguments

  - `γ`: A scalar or vector of step sizes for the random walk.
  - `C`: A positive definite matrix representing the covariance of the proposal distribution.
"""
struct RandomWalkProposal <: Proposal
    γ::Union{Float64,Array{Float64,1}}
    C::PDMat
end

# Constructor for when you only have gamma and dimension
RandomWalkProposal(γ::Union{Float64,Array{Float64,1}}, dim::Int) =
    RandomWalkProposal(γ, PDMat(Matrix(I, dim, dim)))

"""
    pCNProposal(β, C)

A preconditioned Crank-Nicolson (pCN) proposal distribution for MCMC sampling.

# Arguments

  - `β`: A scalar representing the step size for the pCN proposal.
  - `C`: A positive definite matrix representing the covariance of the proposal distribution.
"""
struct pCNProposal <: Proposal
    β::Float64
    C::PDMat
end

function proposal(θ, proposal::RandomWalkProposal)
    return rand(MvNormal(θ, proposal.γ .^ 2 .* proposal.C))
end

function proposal(θ, proposal::pCNProposal)
    return rand(MvNormal(√(1 - proposal.β^2) .* θ, proposal.β^2 .* proposal.C))
end

"""
    proposal(θ, proposal)

Calculate a new proposal from the current parameters `θ` using the specified `proposal`
distribution.
"""
function proposal(θ, proposal::Distribution)
    return θ .+ rand(proposal)
end

export SqExpMvNormal

"""
    SqExpMvNormal(dim, lengthscale, variance)

Define a squared exponential multivariate kernel matrix.

# Arguments

  - `dim`: The dimensionality of the parameter space.
  - `lengthscale`: The lengthscale of the squared exponential kernel, controlling the
                   smoothness of the proposal distribution.
  - `variance`: The variance of the proposal distribution, controlling the overall scale of
                the proposals.
"""
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
    C = p.variance .* exp.(-((xs .- xs') .^ 2) / (p.lengthscale^2))
    return MvNormal(zeros(p.dim), PDMat(Matrix(C)))
end

export MCMCModel

"""
    MCMCModel(posterior, forward, observation, proposal)

Define a model for MCMC sampling, encapsulating the posterior distribution, forward model,
observed data, and proposal distribution.

# Arguments

  - `posterior`: A `Posterior` struct containing the prior and likelihood distributions.
  - `forward`: A function that takes parameters `θ` and returns simulated observations.
  - `observation`: An `ObservationData` struct containing the observed data and sensor information.
  - `proposal`: A `Proposal` struct or a `Distribution` that defines the sampling step.
"""
struct MCMCModel
    posterior::Posterior
    forward::Function
    observation::ObservationData
    proposal::Union{Distribution,Proposal}
end

export logjoint

"""
    logjoint(model::MCMCModel, θ)

Compute the log joint probability of parameters `θ` given the `MCMCModel`.
"""
function logjoint(model::MCMCModel, θ)
    log_prior = sum(logprior(model.posterior, θ))
    if log_prior == -Inf
        return -Inf, 0.0, -Inf
    end

    try
        sim_observations = model.forward(θ)
        log_likelihood =
            sum(loglikelihood(model.posterior, sim_observations .- model.observation.H))
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

"""
    sample_chain(model::MCMCModel, n, initial_θ; verbose=false, logging=Progress(n), burn_in=0)

Run the Metropolis-Hastings MCMC sampling algorithm for a given `MCMCModel`, number of
samples `n`, and initial parameters `initial_θ`.
"""
function sample_chain(
    model::MCMCModel,
    n,
    initial_θ;
    verbose = false,
    logging = Progress(n),
    burn_in = 0,
)
    chain = zeros(n - burn_in + 1, length(initial_θ) + 4)
    θ = initial_θ
    lpost, ll, lp = logjoint(model, θ)
    chain[1, :] = [θ..., lpost, ll, lp, 1.0]
    acceptance_rate = 1.0
    accepted = 1
    for i = 1:n
        θ_new = proposal(θ, model.proposal)
        lpost_new, ll_new, lp_new = logjoint(model, θ_new)

        if (rand()) < exp(lpost_new - lpost)
            accepted += 1
            θ = θ_new
            lpost = lpost_new
            ll = ll_new
            lp = lp_new
        end
        acceptance_rate = accepted / (i + 1)
        if i > burn_in
            chain[i-burn_in+1, :] = [θ..., lpost, ll, lp, acceptance_rate]
        end
        if verbose
            next!(logging, showvalues = [("iteration count", i)])
        end
    end
    return chain
end

function sample_chain(model::MCMCModel, setup::MCMCSetup; kargs...)
    return sample_chain(
        model,
        setup.n,
        setup.init;
        burn_in = setup.burn_in,
        kargs...,
    )
end

function sample_chain(model::MCMCModel, setup::MCMCSetup, initial_θ; kargs...)
    return sample_chain(
        model,
        setup.n,
        initial_θ;
        burn_in = setup.burn_in,
        kargs...,
    )
end
