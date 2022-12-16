

"""
    lambda_ref(lambda) -> lambda_ref

Gives the reference wavelength, given a map of wavelengths .

"""
lambda_ref(lambda::AbstractMatrix{T}) where {T,N} = maximum(lambda)



"""
    build_model(gamma, SED, Interp)

Yields the operator Diag(gamma .* SED) .* Interp.

"""
function build_model(gamma::AbstractArray{T,N},
    SED::AbstractArray{T,N},
    Interp::SparseInterpolator{T,S,N}) where {T,S,N}
    
    return Diag(gamma .* SED) * Interp
end



"""
"""
function build_B_star(gamma::AbstractMatrix{T},
    star_SED::AbstractArray{T,N},
    H_star::SparseInterpolator{T,S,N}) where {T,S,N}
    
    return build_model(gamma, star_SED, H_star)
end

function build_B_star(gamma::AbstractMatrix{T},
    F_star::SparseInterpolator{T,S,N},
    x::AbstractVector{T},
    H_star::SparseInterpolator{T,S,N}) where {T,S,N}
    
    return build_B_star(gamma, F_star*x, H_star)
end

function build_B_star(lambda::AbstractArray{T,N},
    rho::AbstractArray{T,N},
    x::AbstractVector{T},
    Ix::Interpolator{T},
    Iy::Interpolator{T}) where {T,N}

    gamma = lambda_ref(lambda) ./lambda
    F_star = SparseInterpolator(kernel(Ix), lambda, nodes(Ix))
    H_star = SparseInterpolator(kernel(Iy), gamma .* rho, nodes(Iy))
    
    return build_B_star(gamma, F_star, x, H_star)
end



"""
"""
function build_A_star(gamma::AbstractArray{T,N},
    SED_y::AbstractArray{T,N},
    F_star::SparseInterpolator{T,S,N}) where {T,S,N}
    
    return build_model(gamma, SED_y, F_star)
end

function build_A_star(gamma::AbstractArray{T,N},
    H_star::SparseInterpolator{T,S,N},
    y::AbstractVector{T},
    F_star::SparseInterpolator{T,S,N}) where {T,S,N}
    
    return build_A_star(gamma, H_star*y, F_star)
end

function build_A_star(lambda::AbstractArray{T,N},
    rho::AbstractArray{T,N},
    nu_star::AbstractVector{T},
    y::AbstractArray{T},
    Ix::Interpolator{T}) where {T,N}

    gamma = lambda_ref(lambda) ./lambda
    Iy = Interpolator(ker, rho .- nu_star[1], length(y))
    F_star = SparseInterpolator(kernel(Ix), lambda, nodes(Ix))
    H_star = SparseInterpolator(kernel(Iy), gamma .* (rho .- nu_star[1]) , nodes(Iy))
    
    return build_A_star(gamma, H_star, y, F_star)
end



"""
"""
function build_A_comp(gamma::AbstractArray{T,N},
    H_comp::AbstractArray{T,N},
    F_comp::SparseInterpolator{T,S,N}) where {T,S,N}
    
    return build_model(gamma, H_comp, F_comp)
end

function build_A_comp(lambda::AbstractArray{T,N},
    rho::AbstractArray{T,N},
    nu_star::AbstractVector{T},
    nu_comp::AbstractVector{T},
    Iz::Interpolator{T};
    rho_pix::Real = rho_pixel) where {T,N}
    
    gamma = lambda_ref(lambda) ./lambda
    H_comp = off_axis_PSF(lambda, rho .- nu_star[1], nu_comp; rho_pix=rho_pix)
    F_comp = SparseInterpolator(kernel(Iz), lambda, nodes(Iz))
    
    return build_A_comp(gamma, H_comp, F_comp)
end


"""
Warning: must be rho corrected with nu_star
Model of off-axis PSF is a Gaussian

"""
function off_axis_PSF(lambda::AbstractArray{T,N},
    rho::AbstractArray{T,N},
    nu_comp::AbstractVector{T};
    rho_pix::Real = rho_pixel) where {T,N}

    gamma = lambda_ref(lambda) ./lambda
    res_PSF = (gamma .*(rho .- nu_comp[1])) ./nu_comp[2]
    
    return (1/(sqrt(2*π)*nu_comp[2])) .*exp.(-1/2 .*res_PSF.^2) .*rho_pix
end




"""
"""
function residuals!(res::AbstractArray{T,N},
    dat::AbstractArray{T,N},
    lambda::AbstractArray{T,N},
    rho::AbstractArray{T,N},
    nu_star::AbstractVector{T},
    nu_comp::AbstractVector{T},
    y::AbstractVector{T},
    x::AbstractVector{T},
    z::AbstractVector{T};
    rho_pix::Real = rho_pixel) where {T,N}
    
    residuals_comp!(res, dat, lambda, rho, nu_star, y, x)
    
    return residuals_star!(res, res, lambda, rho, nu_star, nu_comp, z; 
                           rho_pix=rho_pix)
end

function residuals(dat::AbstractArray{T,N},
    lambda::AbstractArray{T,N},
    rho::AbstractArray{T,N},
    nu_star::AbstractVector{T},
    nu_comp::AbstractVector{T},
    y::AbstractVector{T},
    x::AbstractVector{T},
    z::AbstractVector{T};
    kwds...) where {T,N}

    return residuals!(vcreate(dat), dat, lambda, rho, nu_star, nu_comp, y, x, z;
                      kwds...)
end


"""
"""
function residuals_star!(res::AbstractArray{T,N},
    dat::AbstractArray{T,N},
    lambda::AbstractArray{T,N},
    rho::AbstractArray{T,N},
    nu_star::AbstractVector{T},
    nu_comp::AbstractVector{T},
    z::AbstractVector{T};
    rho_pix::Real = rho_pixel) where {T,N}

    Iz = Interpolator(ker, lambda, length(z))
    A_comp = build_A_comp(lambda, rho, nu_star, nu_comp, Iz; rho_pix=rho_pix)
    copyto!(res, dat - A_comp*z)

    return res
end

function residuals_star(dat::AbstractArray{T,N},
    lambda::AbstractArray{T,N},
    rho::AbstractArray{T,N},
    nu_star::AbstractVector{T},
    nu_comp::AbstractVector{T},
    z::AbstractVector{T};
    rho_pix::Real = rho_pixel) where {T,N}

    return residuals_star!(vcreate(dat), dat, lambda, rho, nu_star, nu_comp, z; 
                           rho_pix=rho_pix)
end


function residuals_comp!(res::AbstractArray{T,N},
    dat::AbstractArray{T,N},
    lambda::AbstractArray{T,N},
    rho::AbstractArray{T,N},
    nu_star::AbstractVector{T},
    y::AbstractVector{T},
    x::AbstractVector{T}) where {T,N}

    Ix = Interpolator(ker, lambda, length(x))
    A_star = build_A_star(lambda, rho, nu_star, y, Ix)
    copyto!(res, dat - A_star*x)

    return res
end

function residuals_comp(dat::AbstractArray{T,N},
    lambda::AbstractArray{T,N},
    rho::AbstractArray{T,N},
    nu_star::AbstractVector{T},
    y::AbstractVector{T},
    x::AbstractVector{T}) where {T,N}

    return residuals_comp!(vcreate(dat), dat, lambda, rho, nu_star, y, x)
end



