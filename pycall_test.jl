using Pkg
Pkg.activate(".")

using Turing
using PyCall
using Plots
using LinearAlgebra
using Optim

PyCall.python
# Import the SWESolver class from the utils module
@pyimport swe_wrapper as swe
@pyimport swe_wrapper.utils as utils
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
measurement = "./experiment_data/Data_Sensors_orig/With_Bathymetry/Heat1.txt"
mean_bc = "./experiment_data/Data_Sensors_orig/With_Bathymetry/mean_bc.txt"
observation = load_observation_data(file_path)

dataObj = utils.data(measurement)

# lbc1 = swe.leftbc(measurement)
# lbc2 = swe.leftbc(mean_bc)
bathy = swe.rampFunc
gaussian = swe.gaussian_bathymetry
# println(norm(lbc1.f(observation.x) - lbc2.f(observation.x)))
# cost(x) = sum((bathy(observation.x) - gaussian(observation.x, x)).^2)
# res = optimize(cost, [4., 0.1])
# println(Optim.minimizer(res))
# println(Optim.minimum(res))
# plot(observation.x, observation.b)
# plot!(observation.x, gaussian(observation.x, (4., 0.1)))
# display(plot!(observation.x, bathy(observation.x)))
# println(norm(gaussian(observation.x, (4., 0.15)) - bathy(observation.x)))

xbounds = (1.5, 15.)
nx =  100
total_t = 10
tstart = 32.
timestep = 1e-3
timestepstr = "1e-3"
g = 9.81
kappa = 0.2
dealias = 3/2
problemtype = "waterchannel"
problem_bathy = "two_gaussian"
# Create SWE solver and calculate the solution
solver = swe.SWESolver(xbounds, timestep, nx, total_t, tstart=tstart, g=g, kappa=kappa, dealias=dealias,  problemtype=problemtype)
x = solver.domain.x
b = gaussian(x, (5, 0.5)) + gaussian(x, (10.5, 1))
# Use the solver to simulate the shallow water equation with the sampled peak
@time sensor_sim, t, simulation_h, _ = solver.solve(b)
H = simulation_h .+ b'
t = t .+ 32

sensor_obs = hcat(dataObj.f[1](t), dataObj.f[2](t), dataObj.f[3](t)).+ 0.3

error = (sensor_sim .- sensor_obs)
println(size(error))
rel_l2_error = [norm(error[:,i])./ norm(sensor_obs[:,i]) for i in 1:size(error,2)]
println("Relative Error: ", rel_l2_error)
plot_path = "plots/simulation_$(problemtype)_$(problem_bathy)_nx=$(nx)_dt=$(timestepstr)"
println("Save figure in $(plot_path)? (y/o/[n]): ")
s = readline()
if s == "y" || s == "o"
    if s == "o"
        println("Enter path: ")
        plot_path = readline()
    end
    mkpath(plot_path)
    cd(plot_path)
    plot(x,t, H, st=:surface, colorbar_title="h [m]")
    xlabel!("x [m]")
    ylabel!("t [s]")
    zlabel!("h [m]")
    savefig("simulation_h_surface_$(problemtype).png")
    plot(x,t, H, st=:heatmap, colorbar_title="h [m]")
    xlabel!("x [m]")
    ylabel!("t [s]")
    savefig("simulation_h_heatmap_$(problemtype).png")
    for i in 1:3
        plot(t, sensor_obs[:,i], label="sensor $(i+1)")
        plot!(t, sensor_sim[:,i], label="sensor $(i+1) sim")
        title!("Relative L2-Error: $(rel_l2_error[i])")
        savefig("sensor_$(i+1)_observation.png")
    end
    plot(t .+ 32, sensor_sim, label=hcat(["sensor $(i)" for i in 2:4]...), xticks=32:2:42, yticks=([0.294:0.002:0.306;],[0.294:0.002:0.306;]))
    xlabel!("t [s]")
    ylabel!("h [m]")
    savefig("sensor_observation_$(problemtype).png")
end
println(t[end])