using Pkg
Pkg.activate(".")
using TOML
using HDF5
using BathymetryReco
using Distributions
using Plots
using DataInterpolations

cd(@__DIR__)

#TODO pass config file as command line argument
config_file = ARGS[1]
config = TOML.parsefile(config_file)
sim_config = read_simulation_parameters(config["simulation"])

xbounds = sim_config.xbounds
nx =  sim_config.nx
total_t = sim_config.tinterval
tstart = sim_config.tstart
timestep = sim_config.timestep
g = sim_config.g
kappa = sim_config.kappa
dealias = sim_config.dealias
problemtype = sim_config.scenario
problem_bathy = sim_config.bathy_name
bc_file = sim_config.bc_file

solver = swe_solver(sim_config);

x = solver.domain.x

# Set up the bathymetry
bathy_config = read_bathymetry_parameters(config["bathymetry"])
bathy_params = bathy_config.θ
npeaks = bathy_config.n_peaks

bathy = zero(x)

if problem_bathy == "gaussian"
    param_tuples =  length(bathy_params)/2 == npeaks ?
    [[bathy_params[i], bathy_params[i+1]] for i in 1:2:length(bathy_params)] : println("Number of peaks and number of parameters doesn't match!")
    for params in param_tuples
       global bathy += bathymetry(x, params)
    end
elseif  problem_bathy == "exact_bathy"
    global bathy = exp_bathymetry(x)
elseif problem_bathy == "random"
    kernel = SqExpMvNormal(nx, 4, 0.005)
    bathy_dist = MvNormal(kernel)
    equi_x = collect(range(xbounds[1], xbounds[2], nx))
    sample_bathy = rand(bathy_dist)
    global pb = plot(equi_x, sample_bathy, label="sample bathymetry")
    global bathy = PCHIPInterpolation(sample_bathy, equi_x)(x)
else
    println("Requested bathymetry type not available")
end

println("Start Simulation")
H_sensor, t_array, h_array, u_array = solver.solve(bathy, sensor_pos=sim_config.sensor_pos)
println("Simulation Done")
dx = (xbounds[2]-xbounds[1])/nx


io_config = read_io_settings(config["output"])
path = io_config.output_dir

if io_config.save
    mkpath(path)
    sim_name = "$(problemtype)_$(problem_bathy)"
    if problem_bathy == "gaussian"
        sim_name *= "_$(npeaks)_peaks"
    end

    sim_path = joinpath(path,sim_name)
    mkpath(sim_path)
    cd(sim_path)

    if problem_bathy == "gaussian"
        sim_details_path = "$bathy_params"
        sim_details_path = "$bathy_params"
        mkpath(sim_details_path)
        cd(sim_details_path)
        savefig(pb, "sample_bathymetry.png")
        savefig(pb, "sample_bathymetry.pdf")
    end

    println("Storing results in: $(pwd())")

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
            attributes(file)["npeaks"] = npeaks
            attributes(file)["bathy_params"] = bathy_params
        end
    end
    open("simulation_config.toml", "w") do io
        TOML.print(io, config)
    end
end
