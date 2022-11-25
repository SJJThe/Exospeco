

"""
    CalibratedLSSData(d, w, rho, lambda) -> D

Yields a structure `D` containing the data `d` and their respective weights 
`w`, the angular separation `rho` and wavelength `lambda` maps calibrating the
detector.
A `GeoCalib` structure can be given to take into account a `mask` of valid 
pixels. In that case the two following command are equivalent:

```
D = CalibratedLSSData(d, w .* mask, rho, lambda)

G = GeoCalib(rho, lambda, mask)
D = CalibratedLSSData(d, w, G)
```

"""
struct CalibratedLSSData{T<:AbstractFloat,N,D<:AbstractArray{T,N},
                         W<:AbstractArray{T,N},R<:AbstractMatrix{T},
                         L<:AbstractMatrix{T}}
    d::D
    w::W
    rho::R
    lambda::L

    function CalibratedLSSData(d::D,
                               w::W,
                               rho::R,
                               lambda::L) where {T<:Real,N,
                                                     D<:AbstractArray{T,N},
                                                     W<:AbstractArray{T,N},
                                                     R<:AbstractMatrix{T},
                                                     L<:AbstractMatrix{T}}
        @assert axes(d) == axes(w)
        @assert (axes(d,1), axes(d,2)) == axes(rho) == axes(lambda)
        if N == 3
            @warn "Processing of multiframe dataset has not yet been implemented."
        end
        return new{T,N,D,W,R,L}(d, w, rho, lambda)
    end
end

Base.axes(D::CalibratedLSSData) = axes(D.d)
Base.size(D::CalibratedLSSData) = size(D.d)
Base.eltype(D::CalibratedLSSData) = eltype(typeof(D))
Base.eltype(::Type{<:CalibratedLSSData{T}}) where {T} = T
Base.show(io::IO, D::CalibratedLSSData{T}) where {T} = begin
    print(io,"CalibratedLSSData{$T}:")
    print(io,"\n - scientific data `d` : ",typeof(D.d))
    print(io,"\n - weight of data `w` : ",typeof(D.w))
    print(io,"\n - spatial geometric map `rho` : ",typeof(D.rho))
    print(io,"\n - spectral geometric map `lambda` : ",typeof(D.lambda))
end

function CalibratedLSSData(d::AbstractArray{T,N},
    w::AbstractArray{T,N},
    G::GeoCalib{T}) where {T,N}

    return CalibratedLSSData(d, w .* G.mask, G.rho, G.lambda)
end






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

