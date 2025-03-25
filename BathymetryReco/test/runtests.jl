using Test
using BathymetryReco
using Distributions

@testset "Test utils" begin
    obs_data, b = load_observation("../../data/toy_measurement/waterchannel_exact_bathy")

    @test size(obs_data.H) == (length(obs_data.t), length(obs_data.x))
    @test obs_data.tstart == 32.0

    noise_var = 0.01
    x = vec(0:0.1:10)
    clean_signal = reshape(sin.(x), length(x), 1) # has maximum absolute displacement of 1
    noisy_signal = copy(clean_signal)
    add_noise!(noisy_signal, noise_var)
    noise = noisy_signal - clean_signal
    @test std(noise) ≈ noise_var atol=1e-2
end