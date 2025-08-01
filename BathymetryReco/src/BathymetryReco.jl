module BathymetryReco

    using CSV
    using DataFrames
    using DataInterpolations
    using Distributions
    using HDF5
    using LinearAlgebra
    using PDMats
    using ProgressMeter
    using PyCall
    using Statistics
    using TOML

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