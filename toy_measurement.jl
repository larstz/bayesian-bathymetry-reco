using Pkg
Pkg.activate(".")
using Distributions
using TOML
using Dates
using PyCall
using Interpolations
using Plots
using Serialization
# Import the SWESolver class from the utils module
@pyimport swe_wrapper as swe
using HDF5

cd(@__DIR__)

#TODO pass config file as command line argument
config = TOML.parsefile("simulation_config.toml")

path = config["output"]["path"]

mkpath(path)

sim_params = config["simulation"]
xbounds = sim_params["xbounds"]
nx =  sim_params["nx"]
total_t = sim_params["tinterval"]
tstart = sim_params["tstart"]
timestep = sim_params["timestep"]
g = sim_params["g"]
kappa = sim_params["kappa"]
dealias = sim_params["dealias"]
problemtype = sim_params["scenario"]
problem_bathy = sim_params["bathymetry"]

solver = swe.SWESolver(xbounds, timestep, nx, total_t,
                        tstart=tstart, g=g, kappa=kappa, dealias=dealias,
                        problemtype=problemtype)

x = solver.domain.x

# Set up the bathymetry
bathy_config = config["bathymetry"]
bathy_params = bathy_config["parameters"]

bathy = zero(x)

if problem_bathy == "gaussian"
    param_tuples =  length(bathy_params)/2 == bathy_config["npeaks"] ?
    [(bathy_params[i], bathy_params[i+1]) for i in 1:2:length(bathy_params)] : println("Number of peaks and number of parameters doesn't match!")
    for params in param_tuples
       global bathy += swe.gaussian_bathymetry(x, params)
    end
elseif  problem_bathy == "exact_bathy"
    global bathy = swe.rampFunc(x)
else
    println("Requested bathymetry type not available")
end

println("Start Simulation")
H_sensor, t_array, h_array, u_array = solver.solve(bathy, sensor_pos=sim_params["sensor_position"])
println("Simulation Done")
dx = (xbounds[2]-xbounds[1])/nx

if config["output"]["save"]
    sim_name = "$(problemtype)_$(problem_bathy)"
    if problem_bathy == "gaussian"
        sim_name *= "_$(bathy_config["npeaks"])_peaks"
    end
    sim_path = joinpath(path,sim_name)

    mkpath(sim_path)
    cd(sim_path)
    h5open("jl_simulation_data.h5", "w") do file
        file["h"] = h_array
        file["u"] = u_array
        file["H_sensor"] = H_sensor
        file["b_exact"] = bathy
        file["xgrid"] = x
        file["t_array"] = t_array
        attributes(file)["T_N"] = total_t
        attributes(file)["xmin"] = xbounds[1]
        attributes(file)["xmax"] = xbounds[2]
        attributes(file)["tstart"] = tstart
        attributes(file)["dt"] = timestep
        attributes(file)["dx"] = dx
        attributes(file)["g"] = g
        attributes(file)["kappa"] = kappa
        attributes(file)["M"] = nx
        if problem_bathy == "gaussian"
            attributes(file)["npeaks"] = bathy_config["npeaks"]
            attributes(file)["bathy_params"] = bathy_params
        end
    end
    open("simulation_config.toml", "w") do io
        TOML.print(io, config)
    end
end
