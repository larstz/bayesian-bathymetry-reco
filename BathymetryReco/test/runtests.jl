using Test
using BathymetryReco
using Distributions

@testset verbose=true "Test utils" begin
    @testset "load_observation" begin
        obs_data, b = load_observation("./test_data/")

        @test size(obs_data.H) == (length(obs_data.t), length(obs_data.x))
        @test obs_data.tstart == 32.0
    end

    @testset "noise" begin
        noise_var = 0.01
        x = vec(0:0.1:10)
        clean_signal = reshape(sin.(x), length(x), 1) # has maximum absolute displacement of 1
        noisy_signal = copy(clean_signal)
        add_noise!(noisy_signal, noise_var)
        noise = noisy_signal - clean_signal
        @test std(noise) ≈ noise_var atol=1e-2
    end

    @testset "config" begin
        config = load_config("./test_data/test_config.toml")
        @testset "sim params" begin
            sim_params = config.sim_params
            @test sim_params.xbounds == [1.5, 15.0]
            @test sim_params.sensor_pos == [3.5, 5.5, 7.5]
            @test sim_params.timestep == 1e-3
            @test sim_params.nx == 100
            @test sim_params.tend == 10
            @test sim_params.g == 9.81
            @test sim_params.kappa == 0.2
            @test sim_params.dealias == 1.5
            @test sim_params.scenario == "waterchannel"
            @test sim_params.bc_file == "./test_data/mean_bc.txt"
            @test sim_params.bathy_name == "exact_bathy"
        end

        @testset "mcmc params" begin
            mcmc_params = config.mcmc_params
            @test mcmc_params.n == 2
            @test mcmc_params.γ == [0.1, 0.01]
            @test mcmc_params.burn_in == 0
            @test mcmc_params.initial_θ == [[3.5, 0.5], [2.5, 1.5]]
            @test mcmc_params.likelihood_σ == 0.01
        end

        @testset "obs settings" begin
            obs_settings = config.obs_settings
            @test obs_settings.path == "./data/toy_measurement/waterchannel_exact_bathy"
            @test obs_settings.noise_var == 0.05
        end
        @testset "io settings" begin
            io_settings = config.io_settings
            @test io_settings.save == true
            @test io_settings.output_dir == "./data/results"
        end
    end
end

@testset "Test bathymetry" begin
    x = vec(collect(0:0.1:10))
    μ = 5.0
    σ² = 1.0
    scale = 0.2
    y = bathymetry(x, μ, σ², scale)
    @test size(y) == size(x)
    @test y[1] ≈ scale*exp(-1/(σ²+1e-16)*(x[1]-μ)^2) atol=1e-2
    @test y[11] ≈ scale*exp(-1/(σ²+1e-16)*(x[11]-μ)^2) atol=1e-2
    @test y[51] ≈ scale*exp(-1/(σ²+1e-16)*(x[51]-μ)^2) atol=1e-2

    μ₁ = 3.0
    σ²₁ = 1.0
    μ₂ = 7.0
    σ²₂ = 1.0
    y = bathymetry(x, μ₁, σ²₁, μ₂, σ²₂)
    @test size(y) == size(x)
    @test y[1] ≈ 0.2*exp(-1/(σ²₁+1e-16)*(x[1]-μ₁)^2) + 0.2*exp(-1/(σ²₂+1e-16)*(x[1]-μ₂)^2) atol=1e-2
    @test y[11] ≈ 0.2*exp(-1/(σ²₁+1e-16)*(x[11]-μ₁)^2) + 0.2*exp(-1/(σ²₂+1e-16)*(x[11]-μ₂)^2) atol=1e-2
    @test y[51] ≈ 0.2*exp(-1/(σ²₁+1e-16)*(x[51]-μ₁)^2) + 0.2*exp(-1/(σ²₂+1e-16)*(x[51]-μ₂)^2) atol=1e-2

    params = vec(sin.(x))
    y = bathymetry(x, params)
    @test size(y) == size(x)
    @test y[1] == params[1]
    @test y[11] == params[11]
    @test y[51] == params[51]
end

@testset "Test mh" begin
    prior = Uniform(-1, 1)
    likelihood = Normal(0, 1)
    pos = Posterior(prior, likelihood)
    obs = observation_data([1], [1], [1], 1)
    model = mcmc_model(pos, x->x, obs)

    θ = [0.0]
    logp = logjoint(model, θ)
    @test logp ≈ logpdf(prior, θ[1]) + logpdf(likelihood, θ[1]- obs.H[1])

    chain = sample_chain(model, 1000, θ)
    @test size(chain) == (1000, length(θ)+1)


end

@testset "Test swe" begin
    sim_params = load_config("./test_data/test_config.toml").sim_params
    obs_data, b = load_observation("./test_data/")
    sim_observations = simulation([4.0, 0.1, 0.2], sim_params, obs_data)
    @test size(sim_observations) == size(obs_data.H)
    @test sim_observations ≈ obs_data.H atol=1e-2
end