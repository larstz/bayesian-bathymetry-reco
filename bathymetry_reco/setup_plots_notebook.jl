### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ e718afa7-1632-4f27-820b-63d06f61961e
begin
	using Pkg
	Pkg.activate("/DATA/Code/bayesian-inverse-problems")
end

# ╔═╡ 3d6f10d2-dfd8-11ef-056b-676ffbb5ebaf
using Plots, Turing, HDF5, Interpolations

# ╔═╡ 1612b2ad-6415-47a2-83e7-a6717b79d463
dir = "/DATA/Code/bayesian-inverse-problems/plots/setup/"

# ╔═╡ 28a1ea56-a638-43c5-9e93-1d1b20baa185
struct observation_data
	t::Array{Float64}
	x::Array{Float64}
	sensors::Array{Float64}
	h::Array{Float64}
	h_obs::Array{Float64}
	h_obs_n::Array{Float64}
	b::Array{Float64}
end

# ╔═╡ 1990269a-4ac5-48ef-8096-2bdd02a8260f
function load_observation_data(file_path::String)
	h5open(file_path, "r") do file
	   t = read(file["t_array"])
	   x = read(file["xgrid"])
	   h = read(file["h"])
	   b = read(file["b_exact"])
	   H = h .+ b
	   obs_interpolated = LinearInterpolation((t, x), H')
	   sensor_pos = [2., 4., 6., 8.]
	   t_measured = collect(0:0.1:10)
	   observation_H = obs_interpolated.(t_measured, sensor_pos')
	   noise_dist = Normal(0, 0.01)
	   noise = rand(noise_dist, size(observation_H))
	   observation_H_n = observation_H + noise
	   return observation_data(collect(t_measured),x, sensor_pos, H, observation_H, observation_H_n, b)
	end
end


# ╔═╡ cd38400f-7fd3-418a-83fb-6fa9d17196c9
begin
	file_path = "/DATA/Code/bayesian-inverse-problems/data/toy_measurement/simulation_data.h5"
	observation = load_observation_data(file_path)
end

# ╔═╡ ec2378a6-7a39-4341-8925-72bb58784497
begin
	x = observation.x
	H0 = observation.h[:,1]
	H0obs = observation.h_obs_n[1,:]
	sensor = observation.sensors
	b = observation.b

	plot(x, H0, label="H")
	plot!(x, b, label="b", color=:orange)
	scatter!(sensor, H0obs, label="H_obs", color=:green, legend=:right)
	ylabel!("z [m]")
	xlabel!("x [m]")
	title!("Initial setup")
	savefig(dir*"initial_setup.pdf")
end

# ╔═╡ 65acc8f6-2e44-4a01-b32d-916c35bdb0ac
begin
	for i in 1:4
		plot(observation.t, observation.h_obs[:,i], label="H_sim Sensor $i")
		plot!(observation.t, observation.h_obs_n[:,i], label="H_obs Sensor $i")
		xlabel!("t [s]")
		ylabel!("z [m]")
		title!("Sim vs Obs")
		savefig(dir*"sensor_vs_sim_sensor$i.pdf")
	end
end

# ╔═╡ ce183fa2-02f2-4075-99b3-4197f3f08b4a
labels = ["Sensor $i" for i in 1:4]

# ╔═╡ 6f0d9486-9986-4044-9e1f-c267a1558121
begin
	plot(observation.t, observation.h_obs, label = reshape(labels, (1, size(labels)[1])))
	xlabel!("t [s]")
	ylabel!("z [m]")
	title!("Sensorheight without noise")
	savefig(dir*"clean_sensors.pdf")
end


# ╔═╡ 16115ea2-662d-4112-bc26-21f7bb0d8634
["Sensor $i" for i in 1:4]

# ╔═╡ Cell order:
# ╠═e718afa7-1632-4f27-820b-63d06f61961e
# ╠═1612b2ad-6415-47a2-83e7-a6717b79d463
# ╠═3d6f10d2-dfd8-11ef-056b-676ffbb5ebaf
# ╠═28a1ea56-a638-43c5-9e93-1d1b20baa185
# ╠═1990269a-4ac5-48ef-8096-2bdd02a8260f
# ╠═cd38400f-7fd3-418a-83fb-6fa9d17196c9
# ╠═ec2378a6-7a39-4341-8925-72bb58784497
# ╠═65acc8f6-2e44-4a01-b32d-916c35bdb0ac
# ╠═ce183fa2-02f2-4075-99b3-4197f3f08b4a
# ╠═6f0d9486-9986-4044-9e1f-c267a1558121
# ╠═16115ea2-662d-4112-bc26-21f7bb0d8634
