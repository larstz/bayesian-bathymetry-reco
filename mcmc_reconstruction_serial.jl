 # +2 for logp and acceptance rate
for i in 1:mcmc_config.n_chains
    chain = sample_chain(model, mcmc_config, init_θ[i], verbose=true, logging=Progress(mcmc_config.n))
    push!(chains, chain)
end
println("Chains finished \n#############################" )
println(size(chains))
println
if store_exp
    mkpath(target_dir)
    cd(target_dir)

    # store the configuration file for reproducibility
    open("./experiment_config.toml", "w") do io
        TOML.print(io, toml_config)
    end

    pc = plot(;title="Chains", xlabel="Iteration", ylabel="Value", legend=:outerright)
    plp = plot(;title="Chain log p(θ)", xlabel="Iteration", ylabel="Value", legend=:outerright)
    pla = plot(;title="Chain acceptance rate α", xlabel="Iteration", ylabel="Value", legend=:outerright)
    pll = plot(;title="Chain log likelihood", xlabel="Iteration", ylabel="Value", legend=:outerright)
    plprior = plot(;title="Chain log prior", xlabel="Iteration", ylabel="Value", legend=:outerright)
    # Serialize the chain
    for (i, initial_θ) in enumerate(init_θ)
        serialize("chain_$i.jls", chains[i])

        # Plot the chain
        plot!(pc,chains[i][:,1:end-4]; label="") # sampled parameters
        plot!(plp, chains[i][:,end-3]; label="$i: log p(θ)") # log p
        plot!(pll, chains[i][:,end-2]; label="$i: log likelihood") # log likelihood
        plot!(plprior, chains[i][:,end-1]; label="$i: log prior") # log prior
        plot!(pla, chains[i][:,end]; label="$i: α") # acceptance rate
    end
    savefig(pc, "./plots/chain.png")
    savefig(plp, "./plots/logp.png")
    savefig(pll, "./plots/loglikelihood.png")
    savefig(plprior, "./plots/logprior.png")
    savefig(pla, "./plots/acceptance_rate.png")
end