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
#b = observation.b[1:2:end-2]
b = zeros(64)
b[1] = rand(Normal(0,0.01),1)[1]
for i in 2:64
    b[i] = rand(Normal(b[i-1],0.01),1)[1]
end
function create_exponential_decay_matrix(n, decay_factor)
    A = zeros(n, n)
    for i in 1:n
        for j in 1:n
            A[i, j] = exp(-1/decay_factor * abs(i - j))
        end
    end
    return A
end

# Create laplacian operator matrix with dirichlet boundary conditions
function create_laplace_matrix_1d(N)
    # N ist die Anzahl der Gitterpunkte
    L = zeros(Float64, N, N)  # Erstellen einer N×N Matrix

    # Füllen der Matrix mit den Werten für den Laplace-Operator
    for i in 1:N
        if i > 1
            L[i, i - 1] = 1  # Nachbar links
        end
        L[i, i] = -2  # Zentraler Punkt
        if i < N
            L[i, i + 1] = 1  # Nachbar rechts
        end
    end
    return 1/N.*L
end



decay_factor = 100  # Decay factor for off-diagonals
A = create_exponential_decay_matrix(nx, decay_factor)
b = rand(MvNormal(zeros(64), A), 1)

# smoothed prior
L = create_laplace_matrix_1d(nx)
gamma = 0.1
w = rand(MvNormal(zeros(nx), 1), 1)
b = L\(gamma.*w)
b = 0.2.* b/maximum(b)
plot(b)
savefig("b_rand_smoothed.png")
# Create SWE solver and calculate the solution
solver = swe.SWESolver(xbounds, timestep, nx, tend, g, kappa, dealias)
solver.set_initial_conditions()
c = 1.2e-3 + minimum(solver.initial_conditions.H)
println(minimum(solver.initial_conditions.H))
b=ones(64)*c

solver.set_initial_conditions()
println(minimum(solver.initial_conditions.H-b))
# Use the solver to simulate the shallow water equation with the sampled peak
@time simulation_h, _, t = solver.solve(b)
println(t[end])