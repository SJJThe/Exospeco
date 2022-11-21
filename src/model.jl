

"""
"""
lambda_ref(lambda_map::AbstractMatrix{T}) where {T,N} = maximum(lambda_map)



"""
build_model(gamma, SED, Interp)

yields an operator which consists on gamma .* SED.* Interp.

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

function build_B_star(lambda_map::AbstractArray{T,N},
    rho_map::AbstractArray{T,N},
    x::AbstractVector{T},
    Ix::Interpolator{T},
    Iy::Interpolator{T}) where {T,N}

    gamma = lambda_ref(lambda_map) ./lambda_map
    F_star = SparseInterpolator(kernel(Ix), lambda_map, nodes(Ix))
    H_star = SparseInterpolator(kernel(Iy), gamma .* rho_map, nodes(Iy))
    
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

function build_A_star(lambda_map::AbstractArray{T,N},
    rho_map::AbstractArray{T,N},
    nu_star::AbstractVector{T},
    y::AbstractArray{T},
    Ix::Interpolator{T}) where {T,N}

    gamma = lambda_ref(lambda_map) ./lambda_map
    Iy = Interpolator(ker, rho_map .- nu_star[1], length(y))
    F_star = SparseInterpolator(kernel(Ix), lambda_map, nodes(Ix))
    H_star = SparseInterpolator(kernel(Iy), gamma .* (rho_map .- nu_star[1]) , nodes(Iy))
    
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
    nu_z::AbstractVector{T},
    Iz::Interpolator{T}) where {T,N}
    
    gamma = lambda_ref(lambda) ./lambda
    H_comp = off_axis_PSF(lambda, rho .- nu_star[1], nu_z)
    F_comp = SparseInterpolator(kernel(Iz), lambda, nodes(Iz))
    
    return build_A_comp(gamma, H_comp, F_comp)
end


"""
Warning: must be rho corrected with nu_star
#TODO: add more realistic model
Model of off-axis PSF is a Gaussian

"""
function off_axis_PSF(lambda::AbstractArray{T,N},
    rho::AbstractArray{T,N},
    nu_z::AbstractVector{T}) where {T,N}

    gamma = lambda_ref(lambda) ./lambda
    res_PSF = (gamma .*(rho .- nu_z[1])) ./nu_z[2]
    
    return (1/(sqrt(2*π)*nu_z[2])) .*exp.(-1/2 .*res_PSF.^2) .*rho_pixel
end




"""
"""
function residuals!(res::AbstractArray{T,N},
    dat::AbstractArray{T,N},
    lambda_map::AbstractArray{T,N},
    rho_map::AbstractArray{T,N},
    nu_star::AbstractVector{T},
    nu_z::AbstractVector{T},
    y::AbstractVector{T},
    x::AbstractVector{T},
    z::AbstractVector{T}) where {T,N}
    
    residuals_comp!(res, dat, lambda_map, rho_map, nu_star, y, x)
    
    return residuals_star!(res, res, lambda_map, rho_map, nu_star, nu_z, z)
end

function residuals(dat::AbstractArray{T,N},
    lambda_map::AbstractArray{T,N},
    rho_map::AbstractArray{T,N},
    nu_star::AbstractVector{T},
    nu_z::AbstractVector{T},
    y::AbstractVector{T},
    x::AbstractVector{T},
    z::AbstractVector{T}) where {T,N}

    return residuals!(vcreate(dat), dat, lambda_map, rho_map, nu_star, nu_z, y, x, z)
end


"""
"""
function residuals_star!(res::AbstractArray{T,N},
    dat::AbstractArray{T,N},
    lambda_map::AbstractArray{T,N},
    rho_map::AbstractArray{T,N},
    nu_star::AbstractVector{T},
    nu_z::AbstractVector{T},
    z::AbstractVector{T}) where {T,N}

    Iz = Interpolator(ker, lambda_map, length(z))
    A_comp = build_A_comp(lambda_map, rho_map, nu_star, nu_z, Iz)
    copyto!(res, dat - A_comp*z)

    return res
end

function residuals_star(dat::AbstractArray{T,N},
    lambda_map::AbstractArray{T,N},
    rho_map::AbstractArray{T,N},
    nu_star::AbstractVector{T},
    nu_z::AbstractVector{T},
    z::AbstractVector{T}) where {T,N}

    return residuals_star!(vcreate(dat), dat, lambda_map, rho_map, nu_star, nu_z, z)
end


function residuals_comp!(res::AbstractArray{T,N},
    dat::AbstractArray{T,N},
    lambda_map::AbstractArray{T,N},
    rho_map::AbstractArray{T,N},
    nu_star::AbstractVector{T},
    y::AbstractVector{T},
    x::AbstractVector{T}) where {T,N}

    Ix = Interpolator(ker, lambda_map, length(x))
    A_star = build_A_star(lambda_map, rho_map, nu_star, y, Ix)
    copyto!(res, dat - A_star*x)

    return res
end

function residuals_comp(dat::AbstractArray{T,N},
    lambda_map::AbstractArray{T,N},
    rho_map::AbstractArray{T,N},
    nu_star::AbstractVector{T},
    y::AbstractVector{T},
    x::AbstractVector{T}) where {T,N}

    return residuals_comp!(vcreate(dat), dat, lambda_map, rho_map, nu_star, y, x)
end



