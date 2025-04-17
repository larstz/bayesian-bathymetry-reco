export Posterior, logprior, loglikelihood
struct Posterior{T1<:Distribution, T2<:Distribution}
    prior::Union{T1, Array{T1}}
    likelihood::Union{T2, Array{T2}}
end

function logprior(p::Posterior, x)
    return logpdf.(p.prior, x)
end

function loglikelihood(p::Posterior, x)
    return logpdf.(p.likelihood, x)
end

export mcmc_model
struct mcmc_model
    posterior::Posterior
    forward::Function
    observation::observation_data
end

export logjoint
function logjoint(model::mcmc_model , x)
    log_prior = sum(logprior(model.posterior,x))
    if log_prior == -Inf
        return -Inf
    end

    try
        sim_observations = model.forward(x)
        log_likelihood = sum(loglikelihood(model.posterior, sim_observations - model.observation.H))
        return log_prior + log_likelihood
    catch err
        if isa(err, DimensionMismatch) || isa(err, BoundsError)
            println("Dimension mismatch or bounds error in forward model / likelihood.")
            println("Current parameters: ", x)
            return -Inf
        else
            rethrow(err)
        end
    end
end

export sample_chain
function sample_chain(model::mcmc_model, n, initial_θ; verbose=false, log=Progress(n), γ=0.1, burn_in=0)
    chain = zeros(n-burn_in+1, length(initial_θ)+1)
    θ = initial_θ
    logp = logjoint(model, θ)
    chain[1, :] = [θ..., logp]
    for i in 1:n
        θ_new = θ + γ .* rand(Normal(0,1), size(θ))

        logp_new = logjoint(model, θ_new)

        if rand() < exp(logp_new - logp)
            θ = θ_new
            logp = logp_new
        end
        if i > burn_in
            chain[i-burn_in+1, :] = [θ..., logp]
        end
        if verbose
            next!(log)
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