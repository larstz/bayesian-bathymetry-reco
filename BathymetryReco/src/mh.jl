export Posterior, logprior, loglikelihood
struct Posterior{T1<:Distribution, T2<:Distribution}
    prior::Union{T1, Array{T1}}
    likelihood::Union{T2, Array{T2}}
end

function logprior(p::Posterior, θ)
    logp = 0.0
    for lp in p.prior
        logp += sum(logpdf(lp, θ))
    end
    return logp
end

function loglikelihood(p::Posterior, θ)
    return logpdf.(p.likelihood, θ)
end

function loglikelihood(p::Posterior, θ, obs)
    return -length(θ)*0.5 * log(2π) -length(θ)*log(p.likelihood.σ) - 1/ (2 * p.likelihood.σ^2)* sum((θ - obs).^2 )
end

export mcmc_model
struct mcmc_model
    posterior::Posterior
    forward::Function
    observation::observation_data
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
    #β = 0.1
    for i in 1:n

        #temp_proposal = γ .* rand(Normal(0,1), size(θ))
        #θ_new = √(1-β^2) .* θ + β .* temp_proposal #γ .* rand(Normal(0,1), size(θ))
        # pCN, β∈[0,1]
        # θ_new = rand(MvNormal(√(1-β^2) .* θ, PDiagMat(β^2 .* ones(length(θ)))))
        θ_new = θ + γ .* rand(MvNormal(zero(θ),PDiagMat(ones(length(θ)))))
        lpost_new, ll_new, lp_new = logjoint(model, θ_new)

        if (rand()) < exp(lpost_new - lpost)
            accepted += 1
            θ = θ_new
            lpost = lpost_new
        end
        acceptance_rate = accepted / (i+1)
        if i > burn_in
            chain[i-burn_in+1, :] = [θ..., lpost, ll_new, lp_new, acceptance_rate]
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