using Pkg
Pkg.activate(".")

using CSV
using DataFrames
using Statistics

heats = ["data/experiment_data/without_bathymetry/Heat$(i).txt" for i in 1:20]

data = Array{Float64,3}(undef, 10000, 5, 20)
keys = Array{String,1}(undef, 5)
for (i, heat) in enumerate(heats)
    df = CSV.read(heat, DataFrame)
    keys[:] = names(df)
    data[:,:,i] = Matrix(df)
end

mean_data = hcat(mean(data, dims=3)[:, 1:end-1,1] , data[:,end,1])
mean_data_df = DataFrame(mean_data, keys)

CSV.write("data/experiment_data/mean_heat_wob.txt", mean_data_df)