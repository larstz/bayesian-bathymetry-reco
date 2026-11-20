using Pkg
Pkg.activate(".")

using BathymetryReco
using Statistics, CSV, DataFrames

# general setup
xbounds    = [1.5, 15.0]
sensor_pos = [3.5, 5.5, 7.5]
tstart     = 32.0
tinterval  = 10.0
g, kappa, dealias = 9.81, 0.2, 1.5
scenario   = "waterchannel"
dt_obs     = 0.01
nx_inf     = 64
dt_inf = 0.01

nxs = [64,96,128,256]
dts = [5e-5, 1e-3,1e-2]

sim_dir    = "data/model_analysis/"
mkpath(sim_dir)

heat_file(cfg, k) = "data/experiment_data/$(cfg)/Heat$(k).txt"

make_setup(bc_file, dt, nx, name) = SimulationSetup(xbounds, sensor_pos, dt, nx, tstart, tinterval,
                                                    g, kappa, dealias, scenario, bc_file, name)

flat_bathy(x) = zero(x)

observe(cfg, k) = load_observation(heat_file(cfg, k), tstart, tinterval, sensor_id=[2, 3, 4])

rms(x) = sqrt(mean(abs2, x))
l2(x) = sqrt(sum(abs2, x))
rell2(x, o) = l2(x)/l2(o)
mm(x)  = round(x * 1e3, sigdigits=2)

# forward model, returns H at the sensors on the observation grid (every dt_obs)

function sim_file(setup::SimulationSetup)
    cfg  = basename(dirname(setup.bc_file))          # e.g. "without_bathymetry"
    heat = splitext(basename(setup.bc_file))[1]      # e.g. "Heat1"
    return joinpath(sim_dir, "$(setup.bathy_name)_$(cfg)_$(heat)_nx$(setup.nx)_dt$(setup.timestep).txt")
end

function forward(setup::SimulationSetup, bathy::Function; force::Bool=false)
    file = sim_file(setup)
    if isfile(file) && !force
        df = CSV.read(file, DataFrame)
        return Matrix(df[:, ["Sensor2", "Sensor3", "Sensor4"]])
    end
    solver = swe_solver(setup)
    x = solver.domain.x                                  # each solver uses its own grid
    H, _, _, _ = solver.solve(bathy(x), sensor_pos=sensor_pos)
    stride = round(Int, dt_obs / setup.timestep)
    H = H[1:stride:end, :]                               # observation time grid
    t = round.(tstart .+ dt_obs .* (0:size(H, 1)-1), digits=2)
    CSV.write(file, DataFrame(Sensor2=H[:, 1], Sensor3=H[:, 2], Sensor4=H[:, 3], Time=t))
    return H
end

# analysis of flat bed bathymetry for 3 heats
flat_heats = 1:3
r_flat = Dict{Int, Matrix{Float64}}()
for k in flat_heats
    # compare forward model with inference discretization flat to flat
    o = observe("without_bathymetry", k)
    H = forward(make_setup(heat_file("without_bathymetry", k), dt_inf, nx_inf, "flat"), flat_bathy)
    @assert size(H) == size(o.H)
    r_flat[k] = o.H .- H
end

# numerical convergence checks for flat bed bathymetry
bc1  = heat_file("without_bathymetry", 1)
H_c = forward(make_setup(bc1, 0.01, 64,   "flat"), flat_bathy)   # inference settings
H_dt = forward(make_setup(bc1, 1e-3, 64,   "flat"), flat_bathy)   # time step of Angel et al.
H_inf = forward(make_setup(bc1, 1e-2, 96,   "flat"), flat_bathy)   # possible inference
H_dx = forward(make_setup(bc1, 0.01, 128,   "flat"), flat_bathy)
H_dx_dt_bump = forward(make_setup(bc1, 0.001, 128,   "bump"), flat_bathy)
H_2dx = forward(make_setup(bc1, 0.01, 256,   "flat"), flat_bathy)
H_f  = forward(make_setup(bc1, 5e-5, 128,  "flat"), flat_bathy)   # data-generation settings

println("Flat bed analysis:")
for (i, s) in enumerate(sensor_pos)
    println("Sensor at $(s) m:",
            "old error estimation: std per heat = ", [std(r_flat[k][:, i]) for k in flat_heats], " m",
            " rms r_flat per heat = ", [mm(rms(r_flat[k][:, i])) for k in flat_heats], " mm",
            " | rms(r_flat heat 2 - heat 1) = ", mm(rms(r_flat[2][:, i] .- r_flat[1][:, i])), " mm",
            " | rms(H_coarse - H_fine) = ", mm(rms(H_c[:, i] .- H_f[:, i])), " mm",
            " | rms(H_dt0.01 - H_dt0.001) = ", mm(rms(H_c[:, i] .- H_dt[:, i])), " mm",
            " | rms(H_nx64 - H_nx128) = ", mm(rms(H_c[:, i] .- H_dx[:, i])), " mm")

end

# Experimental bathymetry

bump_heats = 1:3
r_bump = Dict{Int, Matrix{Float64}}()
for k in bump_heats
    # compare forward model with inference discretization bump to bump
    o = observe("with_bathymetry", k)
    H = forward(make_setup(heat_file("with_bathymetry", k), dt_inf, nx_inf, "exp"), exp_bathymetry)
    @assert size(H) == size(o.H)
    r_bump[k] = o.H .- H
end

# numerical convergence checks for experimental bathymetry
bc1_bump  = heat_file("with_bathymetry", 1)
H_c_bump  = forward(make_setup(bc1_bump, 0.01, 64,   "bump"), exp_bathymetry)   # inference settings
H_dt_bump = forward(make_setup(bc1_bump, 1e-3, 64,   "bump"), exp_bathymetry)   # time step of Angel et al.
H_inf_bump  = forward(make_setup(bc1_bump, 0.01, 96,   "bump"), exp_bathymetry)   # possible inference settings
H_dx_bump  = forward(make_setup(bc1_bump, 0.01, 128,   "bump"), exp_bathymetry)
H_dx_dt_bump = forward(make_setup(bc1_bump, 0.001, 128,   "bump"), exp_bathymetry)
H_2dx_bump  = forward(make_setup(bc1_bump, 0.01, 256,   "bump"), exp_bathymetry)
H_f_bump  = forward(make_setup(bc1_bump, 5e-5, 128,  "bump"), exp_bathymetry)   # data-generation settings

println("Experimental bathymetry:")
for (i, s) in enumerate(sensor_pos)
    println("Sensor at $(s) m:",
            "old error estimation: std per heat = ", [std(r_bump[k][:, i]) for k in bump_heats], " m",
            " rms r_bump per heat = ", [mm(rms(r_bump[k][:, i])) for k in bump_heats], " mm",
            " | rms(r_bump heat 2 - heat 1) = ", mm(rms(r_bump[2][:, i] .- r_bump[1][:, i])), " mm",
            " | rms(F_coarse - F_fine) = ", mm(rms(H_c_bump[:, i] .- H_f_bump[:, i])), " mm",
            " | rms(F_dt0.01 - F_dt0.001) = ", mm(rms(H_c_bump[:, i] .- H_dt_bump[:, i])), " mm",
            " | rms(F_nx64 - F_nx96) = ", mm(rms(H_c_bump[:, i] .- H_dx_bump[:, i])), " mm",
            " | rms(F_nx128 - F_nx256) = ", mm(rms(H_dx_bump[:, i] .- H_2dx_bump[:, i])), " mm")
end
