using Plots, StatsPlots, StatsBase
using PlutoUI
using Random
using Statistics
using LaTeXStrings
using Optim
using Turing

function u(x,c)
    # Function to evaluate the solution of the advection equation given, x,t,c
	return exp(-0.5*(x[1]-c*x[2]+2)^2)
end

function u0(x)
    # Initial condition
    return exp(-0.5*(x[1]+2)^2)
end

# define our measurement points
t = range(0,1,10)
x = range(-2,2, 20)
x_t = collect(Iterators.product(x,t))#vcat(hcat(x,repeat([t[1]], length(x))), hcat(x,repeat([t[2]], length(x))))
std_v = 0.01
n = std_v*randn(length(x_t))
c = 4

# generate the data
y = [u(x,c) + n for x in x_t]