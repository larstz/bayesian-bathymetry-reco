module BathymetryReco

    using Distributions
    using TOML
    using HDF5
    using LinearAlgebra
    using PyCall
    pushfirst!(PyVector(pyimport("sys")["path"]), "")
    swe = PyNULL()

    function __init__()
        copy!(swe, pyimport("swe_wrapper"))
    end

    include("utils.jl")
    include("bathymetry.jl")
    include("swe.jl")
    include("mh.jl")
end