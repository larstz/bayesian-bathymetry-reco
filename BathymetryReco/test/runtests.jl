using Test
using BathymetryReco
using Distributions

@testset verbose=true "Test utils" begin
    @testset "load_observation" begin
        sensor_rate = 0.1
        obs_data, b = load_toy_observation("./test_data/", sensor_rate=sensor_rate)
        measurement_data = load_observation("./test_data/test_measurement.txt", 32.0, 10.0)
        @test size(obs_data.H) == (length(obs_data.t), length(obs_data.x))
        @test length(obs_data.t) == 10/sensor_rate+1.
        @test size(measurement_data.H) == (length(measurement_data.t), length(measurement_data.x))
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
            @test sim_params.tstart == 32
            @test sim_params.tinterval == 10
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
            @test mcmc_params.n_chains == 10
            @test mcmc_params.dim == 64
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
    x = vec(collect(1.5:0.25:15.5))
    μ = 4.0
    σ² = 0.05
    scale = 0.2
    y = bathymetry(x, μ, σ², scale)
    @test size(y) == size(x)
    @test y[1] ≈ scale*exp(-1/(2*σ²+1e-16)*(x[1]-μ)^2) atol=1e-16
    @test y[11] ≈ scale*exp(-1/(2*σ²+1e-16)*(x[11]-μ)^2) atol=1e-16
    @test y[51] ≈ scale*exp(-1/(2*σ²+1e-16)*(x[51]-μ)^2) atol=1e-16

    μ₁ = 3.0
    σ²₁ = 1.0
    μ₂ = 7.0
    σ²₂ = 1.0
    y2 = bathymetry(x, μ₁, σ²₁, μ₂, σ²₂)
    @test size(y2) == size(x)
    @test y2[1] ≈ 0.2*exp(-1/(2*σ²₁+1e-16)*(x[1]-μ₁)^2) + 0.2*exp(-1/(2*σ²₂+1e-16)*(x[1]-μ₂)^2) atol=1e-16
    @test y2[11] ≈ 0.2*exp(-1/(2*σ²₁+1e-16)*(x[11]-μ₁)^2) + 0.2*exp(-1/(2*σ²₂+1e-16)*(x[11]-μ₂)^2) atol=1e-16
    @test y2[51] ≈ 0.2*exp(-1/(2*σ²₁+1e-16)*(x[51]-μ₁)^2) + 0.2*exp(-1/(2*σ²₂+1e-16)*(x[51]-μ₂)^2) atol=1e-16

    params = vec(sin.(x))
    yp = bathymetry(x, params)
    @test size(y) == size(x)
    @test yp[1] == params[1]
    @test yp[11] == params[11]
    @test yp[51] == params[51]

    y_exp = exp_bathymetry(x)
    @test size(y_exp) == size(x)
    @test y_exp[1] ≈ y[1] atol=1e-16
    @test y_exp[11] ≈ y[11] atol=1e-16
    @test y_exp[51] ≈ y[51] atol=1e-16

end

@testset "Test mh" begin
    prior = [Uniform(-1, 1)]
    likelihood = Normal(0, 1)
    pos = Posterior(prior, likelihood)
    obs = ObservationData([1], [1], [1], [1.], 1., [1.])
    proposal = Normal(0, 0.1)
    model = MCMCModel(pos, x->x, obs, proposal)

    θ = [0.0]
    logp, logl, logpr = logjoint(model, θ)
    @test logp ≈ logpdf(prior[1], θ[1]) + logpdf(likelihood, θ[1]- obs.H[1])
    @test logl ≈ logpdf(likelihood, θ[1]- obs.H[1])
    @test logpr ≈ logpdf(prior[1], θ[1])

    chain = sample_chain(model, 1000, θ)
    @test size(chain) == (1001, length(θ)+4)

end

@testset "Test swe" begin
    sim_params = load_config("./test_data/test_config.toml").sim_params
    real_data = load_observation("./test_data/test_measurement.txt", 32.0, 10.0)
    sim_real = simulation([4.0, 0.05, 0.2], sim_params, real_data)
end