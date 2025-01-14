using Turing
using PyCall

# Import the SWESolver class from the utils module
@pyimport utils as pyutils

# Define the Turing model
@model function shallow_water_model(observation)
    # Define the prior for the bathymetry peak
    peak ~ Normal(0, 1)

    # Create an instance of the SWESolver
    solver = pyutils.SWESolver()

    # Use the solver to simulate the shallow water equation with the sampled peak
    simulation = solver.solve(peak)

    # Compare the simulation with the observation
    observation ~ MvNormal(simulation, 0.1)
end

# Example usage
# Assuming `observation` is your observed data
observation = [1.0, 2.0, 3.0]  # Replace with your actual observation data

# Instantiate the model
model = shallow_water_model(observation)

# Sample from the posterior
chain = sample(model, NUTS(), 1000)

# Print the results
println(chain)