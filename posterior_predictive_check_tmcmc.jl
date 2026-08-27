using Pkg
Pkg.activate(".")
using Serialization
using MCMCChains
using Distributions
using Distributed

parallel = true

if parallel
    addprocs(32)
    @everywhere begin
        using BathymetryReco
    end
else
    using BathymetryReco
end


println("#############################\nRead in chain" )

exp = "data/results/heat_tests/mean_tests/sensor-2-3-4/prior-sparse/proposal-rw/stepsize-0.001/2026-06-02-09-09-56_mean_heat_with_tmcmc" #ARGS[1]
ani = false
include_adjoint = false
plot_sensor_simulation = false
appendix = ""
chain = deserialize(joinpath(exp, "chain_1.jls"))

config = load_config(joinpath(exp, "experiment_config.toml"))
sim_config = config.sim_params
mcmc_config = config.mcmc_params
obs_config = config.obs_settings

# Load the data
if obs_config.real_data
    obs_data = load_observation(obs_config.path, sim_config.tstart, sim_config.tinterval)
else
    obs_data, exact_b = load_toy_observation(obs_config.path, obs_config.noise_var, sensor_rate=obs_config.sensor_rate)
end

obs_H = reshape(obs_data.H, (1, size(obs_data.H)...))

if parallel
    @everywhere forward(params) = simulation(params, $sim_config, $obs_data)
else
    forward(params) = simulation(params, sim_config, obs_data)
end

burnin = 1000
sample_number = size(chain,1) - burnin

bathy = chain[burnin+1:burnin+sample_number,1:mcmc_config.dim]
lp = chain[burnin+1:burnin+sample_number,mcmc_config.dim+1]
ar = chain[burnin+1:burnin+sample_number,end]

sensor_simulations = Array{Float64}(undef, sample_number, length(obs_data.t), length(obs_data.x))

if parallel
    sims = pmap(1:sample_number) do i
        forward(bathy[i, :])
    end
    for (i, sim) in enumerate(sims)
        sensor_simulations[i, :, :] = sim
    end
else
    for i in 1:sample_number
        println("Simulating for sample ", i, " with log-posterior ", lp[i], " and acceptance rate ", ar[i])
        sim = forward(bathy[i, :])
        sensor_simulations[i, :, :] = sim
    end
end

serialize( joinpath(exp, "sensor_simulations.jls"), sensor_simulations)

sim_chains = Chains(sensor_simulations)
serialize(joinpath(exp, "sim_chains.jls"), sim_chains)
