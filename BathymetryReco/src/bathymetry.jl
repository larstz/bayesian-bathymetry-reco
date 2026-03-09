export bathymetry
"""
    bathymetry(x, p, w=1., h=0.2)

Compute the bathymetry at locations `x` given a Gaussian-shaped feature centered at `p` with
width `w` and height `h`.
"""
function bathymetry(x::AbstractArray{T,1}, p::T, w::T = 1.0, h::T = 0.2) where {T<:Real}
    return h * exp.(-1 / (2 * w + 1e-16) .* (x .- p) .^ 2)
end

bathymetry(x::AbstractArray{T,1}, p₁::T, w₁::T, p₂::T, w₂::T) where {T<:Real} =
    bathymetry(x, p₁, w₁) .+ bathymetry(x, p₂, w₂)

"""
    bathymetry(x, params::AbstractArray{T,1})
Compute the bathymetry at locations `x` given a vector of parameters `params`.
If `params` has more than 4 elements, it is interpreted as a discretized bathymetry.
Otherwise, it is interpreted as parameters of one or two Gaussian-shaped features.
"""
bathymetry(x::AbstractArray{T,1}, params::AbstractArray{T,1}) where {T<:Real} =
    length(params) > 4 ? params : bathymetry(x, params...)

export exp_bathymetry
"""
    exp_bathymetry(x)

Compute the bathymetry used in the wave flume experiment at locations `x`.
"""
function exp_bathymetry(x::AbstractArray{T,1}) where {T<:Real}
    return swe.rampFunc(x)
end
