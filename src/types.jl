

"""
"""
struct CalibratedLSSData{T<:Real,M}
    d::AbstractArray{T,M}
    w::AbstractArray{T,M}
    rho_map::AbstractMatrix{T}
    lambda_map::AbstractMatrix{T}

    function CalibratedLSSData{T,M}(d::AbstractArray{T,M},
                                    w::AbstractArray{T,M},
                                    rho_map::AbstractMatrix{T},
                                    lambda_map::AbstractMatrix{T}) where {T<:Real,M}
        @assert axes(d) == axes(w)
        @assert (axes(d,1), axes(d,2)) == axes(rho_map) == axes(lambda_map)
        if M == 3
            @warn "Processing of multiframe dataset has not yet been implemented."
        end
        return new{T,M}(d, w, rho_map, lambda_map)
    end
end
get_data(D::CalibratedLSSData) = D.d
get_weights(D::CalibratedLSSData) = D.w
get_spatial_law(D::CalibratedLSSData) = D.rho_map
get_spectral_law(D::CalibratedLSSData) = D.lambda_map

function CalibratedLSSData(d::AbstractArray{T,M},
    w::AbstractArray{T,M},
    rho_map::AbstractMatrix{T},
    lambda_map::AbstractMatrix{T}) where {T<:Real,M}
    
    return CalibratedLSSData{T,M}(d, w, rho_map, lambda_map)
end

Base.size(D::CalibratedLSSData) = size(get_data(D))
Base.eltype(D::CalibratedLSSData{T,M}) where {T,M} = T
Base.show(io::IO, D::CalibratedLSSData{T,M}) where {T,M} = begin
    print(io,"CalibratedLSSData{$T,$M}:")
    print(io,"\n - scientific data `d` : ",typeof(get_data(D)))
    print(io,"\n - weight of data `w` : ",typeof(get_weights(D)))
    print(io,"\n - spatial geometric map `rho_map` : ",typeof(get_spatial_law(D)))
    print(io,"\n - spectral geometric map `lambda_map` : ",typeof(get_spectral_law(D)))
end




"""
    call!([α::Real=1,] f, x, g; incr::Bool = false)

gives back the value of α*f(x) while updating its gradient in g. incr is a
boolean indicating if the gradient needs to be incremanted or reseted.

"""
call!(f, x, g; kwds...) = call!(1.0, f, x, g; kwds...)
"""
    call([α::Real=1,] f, x)

yields the value of α*f(x).

"""
call(f, x) = call(1.0, f, x)




##### Interpolator
"""
    Interpolator(k, pos)

yields the ingredients to the interpolation operator of kernel k and of
interpolation grid pos.

"""
struct Interpolator{T<:AbstractFloat,K<:Kernel{T},V<:StepRangeLen{T}}
    k::K # interpolation kernel
    pos::V # vector of evenly spaced coordinates
end
kernel(I::Interpolator) = I.k
nodes(I::Interpolator) = I.pos

Base.length(I::Interpolator) = length(nodes(I))
Base.axes(I::Interpolator) = axes(nodes(I))

function Interpolator(k, map::AbstractArray{T,M}, N::Int) where {T,M}
    mmin, mmax = extrema(map)
    pos = range(mmin; length=N, stop=mmax)
    return Interpolator(k, pos)
end




########## Regularization ##########

"""
    HomogeneousRegularization(mu, deg, f)

yields a structure representing an homogeneous regularization and containing
the multiplier which tunes it, its degree and the computing process/function
which gives its value.

"""
struct HomogeneousRegularization{F}
    mu::Float64
    deg::Float64
    f::F # function
end
multiplier(R::HomogeneousRegularization) = R.mu
degree(R::HomogeneousRegularization) = R.deg
func(R::HomogeneousRegularization) = R.f

HomogeneousRegularization(mu::Real, f) = HomogeneousRegularization(mu, degree(f), f)
HomogeneousRegularization(mu::Real, R::HomogeneousRegularization) = HomogeneousRegularization(mu, degree(R), func(R))
HomogeneousRegularization(f) = HomogeneousRegularization(1.0, f)


Base.:(*)(a::Real, R::HomogeneousRegularization) =
    HomogeneousRegularization(a*multiplier(R), degree(R), func(R))
Base.show(io::IO, R::HomogeneousRegularization) = begin
    print(io,"HomogeneousRegularization:")
    print(io,"\n - level `mu` : ",multiplier(R))
    print(io,"\n - degree `deg` : ",degree(R))
    print(io,"\n - function `func` : ",func(R))
end


function call!(α::Real,
               R::HomogeneousRegularization,
               x::AbstractArray{T,N},
               g::AbstractArray{T,N};
               incr::Bool = false) where {T,N}
    return call!(α*multiplier(R), func(R), x, g; incr=incr)
end

function call!(R::HomogeneousRegularization,
               x::AbstractArray{T,N},
               g::AbstractArray{T,N};
               incr::Bool = false) where {T,N}
    return call!(multiplier(R), func(R), x, g; incr=incr)
end

function call(α::Real,
              R::HomogeneousRegularization,
              x::AbstractArray{T,N}) where {T,N}
    return call(α*multiplier(R), func(R), x)
end

function call(R::HomogeneousRegularization,
              x::AbstractArray{T,N}) where {T,N}
    return call(multiplier(R), func(R), x)
end


"""
    QuadraticSmoothness()

yields an instance of Tikhonov smoothness regularization.

"""
struct QuadraticSmoothness end

function call!(α::Real,
               ::QuadraticSmoothness,
               x::AbstractArray{T,N},
               g::AbstractArray{T,N};
               incr::Bool = false) where {T,N}
    D = Diff()
    apply!(2*α, D'*D, x, (incr ? 1 : 0), g)
    return Float64(α*vnorm2(D*x)^2)
end

function call(α::Real,
              ::QuadraticSmoothness,
              x::AbstractArray{T,N}) where {T,N}
    D = Diff()
    return Float64(α*vnorm2(D*x)^2)
end

degree(::QuadraticSmoothness) = 2.0

const quadraticsmoothness = HomogeneousRegularization(QuadraticSmoothness())






"""
    SubProblem(A, b, w, R)

yields a structure containing the ingredients to compute a sub-problem
criterion for a value of x:
                (A.x - b)'.Diag(w).(A.x - b) + R(x)
with R an homogeneous regularization structure.

"""
struct SubProblem{T,N,M<:Mapping,B<:AbstractArray{T,N},W<:AbstractArray{T,N},C<:HomogeneousRegularization}
    A::M
    b::B
    w::W
    R::C
end

function call!(α::Real,
               S::SubProblem{T},
               x::AbstractArray{T,N},
               g::AbstractArray{T,N};
               incr::Bool = false) where {T,N}
    A, b, w, R = S.A, S.b, S.w, S.R
    res = A*x - b
    wres = w .*res
    lkl = α*vdot(res, wres)
    apply!(2*α, LazyAlgebra.Adjoint, A, wres, true, (incr ? 1 : 0), g)# g = (0 : 1)*g + 2*A'*wres
    return Float64(lkl + call!(α, R, x, g; incr=true))
end

function call(α::Real,
              S::SubProblem{T,N},
              x::AbstractArray{T,N}) where {T,N}
    A, b, w, R = S.A, S.b, S.w, S.R
    res = A*x - b
    lkl = α*vdot(res, w, res)
    return Float64(lkl + call(α, R, x))
end

