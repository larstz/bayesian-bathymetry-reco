export Posterior, logprior, loglikelihood
struct Posterior
    prior::Distribution
    likelihood::Distribution
end

function logprior(p::Posterior, x)
    return logpdf(p.prior, x)
end

function loglikelihood(p::Posterior, x)
    return logpdf(p.likelihood, x)
end

export mcmc_model
struct mcmc_model
    posterior::Posterior
    forward::Function
    observation::observation_data
end

export logjoint
function logjoint(model::mcmc_model , x)
    log_prior = logprior(model.posterior,x)
    if log_prior == -Inf
        return -Inf
    end
    sim_observations = model.forward(x)
    try
        log_likelihood = sum(loglikelihood(model.posterior, sim_observations - observation.H))
        return log_prior + log_likelihood
    catch err
        if isa(err, DimensionMismatch)
            return -Inf
        else
            rethrow(err)
        end
    end
end

export sample
function sample(model::mcmc_model, n, initial_x; γ=0.1, burn_in=0)
    chain = zeros(n-burn_in, length(initial_x)+1)
    x = initial_x
    logp = logjoint(model, x)
    for i in ProgressBar(1:n)
        x_new = x + γ .* rand(Normal(0,1), size(x))

        logp_new = logjoint(model, x_new)

        if rand() < exp(logp_new - logp)
            x = x_new
            logp = logp_new
        end
        if i > burn_in
            chain[i-burn_in, :] = [x..., logp]
        end
    end
    return chain
end

function sample(model::mcmc_model, setup::mcmc_setup)
    return sample(model, setup.n, setup.init, γ=setup.γ, burn_in=setup.burn_in)
end
