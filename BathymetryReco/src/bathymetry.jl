export bathymetry
function bathymetry(x::Array{Float64,1}, μ::Float64, σ²::Float64=1., scale::Float64=0.2)
    return scale * exp.(-1/(σ²+1e-16) .*(x .- μ).^2)
end

bathymetry(x::Array{Float64,1}, μ₁::Float64, σ²₁::Float64,μ₂::Float64, σ²₂::Float64) = bathymetry(x, μ₁, σ²₁) .+ bathymetry(x, μ₂, σ²₂)
bathymetry(x::Array{Float64,1}, params::Array{Float64,1}) = length(params)>4 ? params : bathymetry(x, params...)

export exp_bathymetry
function exp_bathymetry(x::Array{Float64,1})
    return swe.rampFunc(x)
end