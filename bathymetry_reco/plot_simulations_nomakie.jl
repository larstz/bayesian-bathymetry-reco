using Pkg
Pkg.activate(".")
Pkg.instantiate()
using HDF5
using Plots
using TOML

sim_name = "data/toy_measurement/waterchannel_exact_bathy"
cd(sim_name)
struct simulation
    x::Array{Float64, 1}
    t::Array{Float64, 1}
    bathymetry::Array{Float64, 1}
    H::Array{Float64, 2}
    H_sensor::Array{Float64, 2}
    sensor_pos::Array{Float64, 1}
end

# Load the simulation data
data = h5open("jl_simulation_data.h5") do file
    # Load the bathymetry
    bathymetry = read(file, "b_exact")
    # Load the measurements
    H_sensor = read(file, "H_sensor")
    # Load the waterlevel
    H = read(file, "h") .+ bathymetry'
    # Load the time
    t = read(file, "t_array")
    # Load the xgrid
    x = read(file, "xgrid")
    # Load the sensor position
    config = TOML.parsefile("simulation_config.toml")
    sensor_pos = config["simulation"]["sensor_position"]
    simulation(x, t, bathymetry, H, H_sensor, sensor_pos)
end

# Plot initial conditions
p = plot(data.x, data.H[1, :], color = :midnightblue, linewidth = 2, label = "h(x, 0)")
plot!(p, data.x, data.bathymetry, color = :brown, linewidth = 2, label = "b(x)")
vline!(p, data.sensor_pos, color = :black, linestyle = :dash, linewidth = 1, label="")
scatter!(p, data.sensor_pos, data.H_sensor[1, :], color = :green, label = "sensor")
xlabel!("x [m]")
ylabel!("z [m]")
title!("Initial waterlevel")
plot!(p, legend=:right)
savefig(p,"initial_conditions_plots.pdf")

# # Plot the sensors
p = plot(data.t, data.H_sensor, linewidth = 2, label = reshape(["Sensor $i" for i in 2:4], 1, 3))
xlabel!("t [s]")
ylabel!("z [m]")
title!("Sensor positions")
plot!(p, legend=:topleft)
savefig(p,"sensor_positions_plots.pdf")
# fig = Figure()
# ax1 = Axis(fig[1, 1], xlabel = "x [m]", ylabel = "z [m]", title = "Sensor positions")
# series!(ax1, data.t, data.H_sensor', linewidth = 2, label = ["Sensor $i" for i in 2:4])
# axislegend(ax1, position = :lt)
# save("sensor_positions.png", fig)

# # Plot the waterlevel
p = surface(data.x, data.t, data.H, colormap = :viridis, xlabel = "x [m]", ylabel = "t [s]", zlabel="z [m]", title = "Waterlevel")
savefig(p,"waterlevel_plots.pdf")
# fig = Figure()
# ax1 = Axis3(fig[1, 1], ylabel = "x [m]", xlabel = "t [s]", zlabel="z [m]", title = "Waterlevel")
# surface!(ax1, data.x, data.t, data.H', colormap = :viridis, interpolate=false)
# save("waterlevel.png", fig)