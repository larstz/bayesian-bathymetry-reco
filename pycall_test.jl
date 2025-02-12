using Pkg
Pkg.activate(".")

using Turing
using PyCall
using Plots

PyCall.python
# Import the SWESolver class from the utils module
@pyimport swe_wrapper as swe
using HDF5 # Has to imported after my module otherwise pycall cannot find swe_wrapper
struct observation_data
    t::Array{Float64}
    x::Array{Float64}
    h::Array{Float64}
    b::Array{Float64}
end

function load_observation_data(file_path::String)
    h5open(file_path, "r") do file
        t = read(file["t_array"])
        x = read(file["xgrid"])
        h = read(file["h"])
        b = read(file["b_exact"])

        return observation_data(t,x,h, b)
    end
end

file_path = "./data/toy_measurement/simulation_data.h5"
observation = load_observation_data(file_path)

xbounds = (0., 10.)
nx =  64
tend = 10
timestep = 1e-3
g = 9.81
kappa = 0.2
dealias = 3/2
bathy_peak = (5,1) #rand(MvNormal(zeros(130), 1),1)
b = observation.b[2:2:end-1]
problemtype = "periodic"

# Create SWE solver and calculate the solution
solver = swe.SWESolver(xbounds, timestep, nx, tend, g, kappa, dealias, problemtype=problemtype)

# Use the solver to simulate the shallow water equation with the sampled peak
@time simulation_h, _, t = solver.solve(b)
H = simulation_h .+ b'
x = solver.domain.x

display(plot(x,t, H, st=:surface, aspect_ratio=1))

println("Save figure? (y/n): ")
s = readline()
if s == "y"
    plot(x,t, H, st=:surface, colorbar_title="h [m]")
    xlabel!("x [m]")
    ylabel!("t [s]")
    zlabel!("h [m]")
    savefig("simulation_h_surface_$(problemtype).png")
    plot(x,t, H, st=:heatmap, colorbar_title="h [m]")
    xlabel!("x [m]")
    ylabel!("t [s]")
    savefig("simulation_h_heatmap_$(problemtype).png")
end
println(t[end])