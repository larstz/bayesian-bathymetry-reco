module BathymetryReco

    using Distributions
    using TOML
    using ProgressBars
    using HDF5
    using PyCall
    swe = pyimport("swe_wrapper")


    export swe
    include("utils.jl")
    include("swe.jl")
    include("mh.jl")

end