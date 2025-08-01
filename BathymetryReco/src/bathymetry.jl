export bathymetry
function bathymetry(x::AbstractArray{T,1}, μ::T, σ²::T=1., scale::T=0.2) where {T<:Real}
    return scale * exp.(-1/(2*σ²+1e-16) .*(x .- μ).^2)
end

bathymetry(x::AbstractArray{T,1}, μ₁::T, σ²₁::T,μ₂::T, σ²₂::T) where {T<:Real} = bathymetry(x, μ₁, σ²₁) .+ bathymetry(x, μ₂, σ²₂)
bathymetry(x::AbstractArray{T,1}, params::Array{T,1}) where {T<:Real} = length(params)>4 ? params : bathymetry(x, params...)

export exp_bathymetry
function exp_bathymetry(x::AbstractArray{T,1}) where {T<:Real}
    return swe.rampFunc(x)
end