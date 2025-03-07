using Test
using BathymetryReco

@testset "Test bathymetry reconstruction" begin
    # Load the observation data
    obs_data, b = load_observation("../../data/toy_measurement/waterchannel_exact_bathy")

    @test size(obs_data.H) == (length(obs_data.t), length(obs_data.x))
    @test obs_data.tstart == 32.0
end