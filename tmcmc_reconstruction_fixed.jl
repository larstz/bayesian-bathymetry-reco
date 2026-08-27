###############################################################################
# MCMC reconstruction of bathymetry from water surface height observations    #
# This script runs multiple MCMC chains in serial and stores the results      #
# along with diagnostic plots.                                                #
#                                                                             #
#   Author: Lars Stietz                                                       #
###############################################################################

using Pkg
Pkg.activate(".")
Pkg.instantiate()
using Distributed, SlurmClusterManager

addprocs(SlurmManager())
@assert nworkers() == parse(Int, ENV["SLURM_NTASKS"]) "Something went wrong with the number of processes!"

using Dates
using TOML
using Serialization
using Plots
using PDMats
using LinearAlgebra

@everywhere begin
    using Distributions
    using BathymetryReco
    using Random

    Random.seed!(1910)
end

ENV["GKSwstype"]="nul"

###############################################################################
# Load the configuration                                                      #
###############################################################################

println("#############################\nRead in config file" )
if isempty(ARGS)
    config_file = abspath("./paper_configs/tmcmc/parameterized_uniform_prior_config.toml")
else
    config_file = abspath(ARGS[1])
end

toml_config = TOML.parsefile(config_file) # load toml to modify later
config = load_config(toml_config)
sim_config = config.sim_params
mcmc_config = config.mcmc_params
obs_config = config.obs_settings
prior_settings = mcmc_config.prior
proposal_settings = mcmc_config.proposal
io_config = config.io_settings



println("##############################\nLoad experiment data")

###############################################################################
# Load the observation data                                                   #
###############################################################################

if obs_config.real_data
    obs_data = load_observation(obs_config.path, sim_config.tstart, sim_config.tinterval,
    sensor_id  = obs_config.sensor_id, noise_var=obs_config.noise_var)
    exp_type = "heat_tests"
    if occursin("mean", obs_config.path)
        exp_type = "heat_tests/mean_tests"
    end
else
    obs_data, exact_b = load_toy_observation(obs_config.path, obs_config.noise_var,
    sensor_rate=obs_config.sensor_rate, sensor_id=obs_config.sensor_id)
    exp_type = "toy_tests"
end

# Set up directory for storing results
store_exp = io_config.save
exp_name = split(splitpath(obs_config.path)[end], ".")[1]
# Directory structure for storing results experiment/sensors/prior/proposal/stepsize/timestamp_expname
target_dir = joinpath(io_config.output_dir,
                      exp_type,
                      "sensor-"*join(obs_config.sensor_id, "-"),
                      "prior-"*join(prior_settings.type,"-"),
                      "proposal-"*proposal_settings.type * "-" * proposal_settings.kernel,
                      "stepsize-"*join(string.(mcmc_config.γ),"-"),
                      "$(Dates.format(now(), "Y-mm-dd-HH-MM-SS"))_$(exp_name)_tmcmc")

println("Storing results in: $target_dir")

println("#############################")

# create plot of the observation signal
ps = plot(;title="Observation signal", xlabel="time [s]", ylabel="Water surface height [m]")
plot!(ps, obs_data.t, obs_data.H; label=reshape(["Sensor $i" for i in obs_config.sensor_id], 1,length(obs_config.sensor_id)))

###############################################################################
# Setup the forward model, likelihood, prior and proposal for MCMC sampling   #
###############################################################################

# define forward model
@everywhere forward_model(params) = simulation(params, $sim_config, $obs_data)

# Defining likelihood distribution
likelihood_σ = mcmc_config.likelihood_σ
if likelihood_σ == 0.0
    flat_signal = forward_model(zeros(mcmc_config.dim))
    likelihood_σ = vec(std(obs_data.H .- flat_signal, dims=1)) # set to std of flat signal residuals
    println("Calculated likelihood std from flat signal residuals: $(likelihood_σ)")
end

println("Using $(likelihood_σ) std for Likelihood distribution.")
likelihood_dist = MvNormal(zeros(size(likelihood_σ)), PDiagMat(likelihood_σ.^2))

# define prior distributions
prior_dist = Vector{Distribution}()
for (i, prior_type) in enumerate(mcmc_config.prior.type)
    prior_param = [mcmc_config.prior.loc[i], mcmc_config.prior.scale[i]]
    if prior_type == "normal"
        push!(prior_dist, Normal(prior_param...))
    elseif prior_type == "uniform"
        push!(prior_dist, Uniform(prior_param...))
    else
        error("Unsupported prior type: $prior_type")
    end
end
println("Using prior distribution: $(prior_settings.type)")

# add newly calculated information to config
toml_config["sampler"]["likelihood_var"] = likelihood_σ

# Put everything into the MCMC model
pos = Posterior(prior_dist, likelihood_dist)
# proposal only needed to fit with the MCMCModel type but not needed in TMCMC
proposal = Normal(0.0,0.0)
model = MCMCModel(pos, forward_model, obs_data, proposal)

# Define initial parameters
n_doubles = Int(mcmc_config.n/length(mcmc_config.initial_θ))
init_θ = reduce(hcat, [v for v in mcmc_config.initial_θ, _ in 1:n_doubles])
init_θ = Matrix(transpose(init_θ)) 
    

toml_config["sampler"]["init"] = init_θ
inip = scatter(init_θ[:,1], init_θ[:,2], label="Initial samples n=$(size(init_θ, 1))")
display(inip)

println("#############################")


###############################################################################
# Run the MCMC sampling                                                       #
###############################################################################
println("Start TMCMC with $(mcmc_config.n) samples: \n#############################" )

time_stat = @timed begin
    final_parameters, S, nEval, ESS = transitional_mcmc(model, mcmc_config, init_θ, verbose=false, parallel_eval=true, collect_evals=false)
end

println("TMCMC finished \n#############################" )

###############################################################################
# Store the chains and create diagnostic plots                                #
###############################################################################
nW = nprocs()

rmprocs(workers())
import StatsPlots
using JLD

if store_exp
    mkpath(target_dir)
    mkpath(joinpath(target_dir,"plots"))
    cd(target_dir)

    # store the configuration file for reproducibility
    # need to change toml_config["sampler"]["init"] from N × Nₚ Matrix to N-vectors
    toml_config["sampler"]["init"] = Vector.(collect(eachrow(toml_config["sampler"]["init"])))
    open("./experiment_config.toml", "w") do io
        TOML.print(io, toml_config)
    end

    # save timings
    time_dict = Dict("time" => time_stat.time, "gctime" => time_stat.gctime, "bytes" => time_stat.bytes, "compile_time" => time_stat.compile_time,
                    "recompile_time" => time_stat.recompile_time, "lock_conflicts" => time_stat.lock_conflicts, "nprocs" => nW-1, "nEval" => nEval)
    open("./timings.toml", "w") do io
        TOML.print(io, Dict("time_summary" => time_dict))
    end

    # save samples
    @save "./final_parameters.jld" final_parameters

    savefig(inip, "initial_samples.png")
    scatter!(inip,final_parameters[:,1], final_parameters[:,2], label="Final samples", title="Final samples", xlabel="Parameter 1", ylabel="Parameter 2")  
    savefig(inip, "final_samples.png")

end

