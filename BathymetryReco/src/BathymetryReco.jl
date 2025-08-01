module BathymetryReco

    using Distributions
    using Statistics
    using TOML
    using HDF5
    using DataFrames
    using CSV
    using LinearAlgebra
    using DataInterpolations
    using PyCall
    using ProgressMeter

    swe = PyNULL()

    function __init__()
        pushfirst!(pyimport("sys")."path", @__DIR__)
        copy!(swe, pyimport("swe_wrapper"))
    end

    include("utils.jl")
    include("bathymetry.jl")
    include("swe.jl")
    include("mh.jl")
end