

"""
"""
function solve!(x0::AbstractVector{T},
    A::Mapping,
    dat::AbstractArray{T,N},
    wgt::AbstractArray{T,N},
    Reg::HomogeneousRegularization;
    nonnegative::Bool = false,
    kwds...) where {T,N}

    @assert is_linear(A)
    S = SubProblem(A, dat, wgt, Reg)

    function fg_solve!(x, g)
        return call!(S, x, g)
    end
    
    if nonnegative
        vmlmb!(fg_solve!, x0; lower=T(0), kwds...)
    else
        vmlmb!(fg_solve!, x0; kwds...)
    end
    
    return x0
end



"""
"""
function best_alpha(x::AbstractVector{T},
    Rx::HomogeneousRegularization,
    y::AbstractVector{T},
    Ry::HomogeneousRegularization) where {T}
    q_x = degree(Rx)
    q_y = degree(Ry)
    
    return ((q_y*call(Ry, y))/(q_x*call(Rx, x)))^(1/(q_x + q_y))
end




"""
fit the geometric parameter of the wavelength and angular separation maps.
"""
function fit_nu_star!(nu_star::AbstractVector{T},
    dat::AbstractArray{T,N},
    wgt::AbstractArray{T,N},
    gamma::AbstractArray{T,N},
    rho::AbstractArray{T,N},
    SED_x::AbstractArray{T,N},
    y::AbstractArray{T};
    nu_box::AbstractVector{T} = [-10mas, 10mas],
    study::Val = Val(false)) where {T,N}

    gamma_nu = gamma
    SED_x_nu = SED_x

    function f(nu)
        rho_nu = rho .- nu[1]
        Iy = Interpolator(ker, rho_nu, length(y))
        H_star_nu = SparseInterpolator(kernel(Iy), gamma .* rho_nu, nodes(Iy))
        res = gamma_nu .* (H_star_nu*y) .* SED_x_nu - dat
        score = vdot(res, wgt, res)
        return score
    end
    if study === Val(true)
        return f
    end

    vcopy!(nu_star, [Brent.fmin(f, nu_box[1], nu_box[2]; rtol=1e-3)[1]])

    return nu_star
end


""" 
"""
#TODO: mettre rho_pixel en mot clef
function fit_nu_z!(nu_z::AbstractVector{T},
    dat::AbstractArray{T,N},
    wgt::AbstractArray{T,N},
    lambda::AbstractArray{T,N},
    rho::AbstractArray{T,N},
    SED_z::AbstractArray{T,N};
    rho_pix::Real = rho_pixel,
    nu_box::AbstractVector{T} = [50.0, 2.0],
    study::Val = Val(false)) where {T,N}

    gamma = lambda_ref(lambda) ./ lambda

    function f(nu)
        H_comp_nu = off_axis_PSF(lambda, rho, nu; rho_pix=rho_pix)
        res = gamma .* H_comp_nu .* SED_z - dat
        return vdot(res, wgt, res)
    end
    if study === Val(true)
        return f
    end
    
    nu0 = copy(nu_z)
    nu_bnd_min = [nu0[1]-nu_box[1], 1.0] .* rho_pix#nu0 .+ [-nu_box[1], 1.0] .* rho_pixel
    nu_bnd_max = nu0 .+ nu_box .* rho_pix
    vcopy!(nu_z, Bobyqa.minimize(f, nu0, nu_bnd_min, nu_bnd_max,
                                 1*rho_pix, 1e-3*rho_pix)[2])#; scale=[])[2])#

    return nu_z
end

