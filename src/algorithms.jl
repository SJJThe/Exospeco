

"""
"""
function exospeco!(x::AbstractVector{T},
    y::AbstractVector{T},
    nu_star::AbstractVector{T},
    z::AbstractVector{T},
    nu_comp::AbstractVector{T},
    dat::AbstractArray{T,N},
    wgt::AbstractArray{T,N},
    rho::AbstractArray{T,N},
    lambda::AbstractArray{T,N},
    Rx::HomogeneousRegularization,
    Ry::HomogeneousRegularization,
    Rz::HomogeneousRegularization;
    auto_calib::Bool = false,
    delay_auto_calib::Bool = false,
    nonnegative::Bool = false,
    maxiter::Integer = 10,
    alphatol::Float64 = 0.1,
    z_tol::Real = 3.0,
    alg_tol::Tuple{Real,Real} = (0.0,1e-3),
    nu_star_box::AbstractVector{T} = [-2rho_pixel, 2rho_pixel],
    nu_z_box::AbstractVector{T} = [50.0, 2.0],
    wordy::Bool = false,
    solvekwds...) where {T,N}

    wordy && println("++ Exospeco method")
    delay_auto_calib && (auto_calib = false)

    # Mask companion for first iteration
    mask_z = ones(T, size(dat))
    loc_z = lambda_ref(lambda) ./ lambda .* abs.(rho .- nu_star[1] .- nu_comp[1])
    mask_z[loc_z .< z_tol*nu_comp[2]] .= T(0)

    res = similar(dat)
    wgt_prime = similar(wgt)
    iter = 0
    res_loss = zeros(size(dat))
    loss = 0.0
    loss_past = 0.0
    while true
        # Hide companion at first iteration
        if iter == 0
            @inbounds for i in eachindex(wgt_prime, mask_z, wgt)
                wgt_prime[i] =  mask_z[i]*wgt[i]
            end
        elseif iter == 1
            copyto!(wgt_prime, wgt)
        end

        # Residuals r_star
        residuals_star!(res, dat, lambda, rho, nu_star, nu_comp, z)
        # Update stellar leakage model
        fitStar!(x, y, nu_star, res, wgt_prime, rho, lambda, Rx, Ry;
                 auto_calib=auto_calib, nonnegative=nonnegative, maxiter=maxiter, 
                 alphatol=alphatol, tol=alg_tol, nu_box=nu_star_box, wordy=wordy, 
                 solvekwds...)
        
        # Residuals r_comp
        residuals_comp!(res, dat, lambda, rho, nu_star, y, x)
        # Update companion model
        fitCompanion!(z, nu_comp, res, wgt, rho, lambda, nu_star, Rz;
               auto_calib=auto_calib, nonnegative=nonnegative, maxiter=maxiter, 
               tol=alg_tol, nu_box=nu_z_box, wordy=wordy, solvekwds...)
        
        # stop criterions
        iter += 1
        residuals!(res_loss, dat, lambda, rho, nu_star, nu_comp, x, y, z)
        loss = vdot(wgt, res_loss, res_loss) + call(Rx, x) + call(Ry, y) + call(Rz, z)
        if iter ≥ maxiter || (abs(loss - loss_past) <= max(alg_tol[1], alg_tol[2]*abs(loss_past)))
            if delay_auto_calib && !auto_calib
                wordy && println("++ Start autocalib")
                auto_calib = true
            else
                break
            end
        end
        loss_past = loss
    end

        return x, y, nu_star, z, nu_comp
end

function exospeco(x0::AbstractVector{T},
    y0::AbstractVector{T},
    nu_star0::AbstractVector{T},
    z0::AbstractVector{T},
    nu_comp0::AbstractVector{T},
    D::CalibratedLSSData{T,N},
    mu_x::Real,
    mu_z::Real;
    kwds...) where {T,N}

    Rx = mu_x*quadraticsmoothness
    Ry = quadraticsmoothness
    Rz = mu_z*quadraticsmoothness
    
    return exospeco!(copy(x0), copy(y0), copy(nu_star0), copy(z0), copy(nu_comp0),
                     D.d, D.w, D.rho, D.lambda, Rx, Ry, Rz; kwds...)
end



"""
"""
function fitStar!(x::AbstractVector{T}, # stellar SED
    y::AbstractVector{T}, # stellar on-axis PSF
    nu_star::AbstractVector{T},
    dat::AbstractArray{T,N}, # residual data Eq. (29d)
    wgt::AbstractArray{T,N},
    rho::AbstractArray{T,N},
    lambda::AbstractArray{T,N},
    Rx::HomogeneousRegularization,
    Ry::HomogeneousRegularization;
    auto_calib::Bool = false,
    nonnegative::Bool = false,
    alpha::Float64 = 1.0,
    maxiter::Integer = 10,
    alphatol::Float64 = 0.1,
    tol::Tuple{Real,Real} = (0.0, 1e-6),
    nu_box::AbstractVector{T} = [-2rho_pixel, 2rho_pixel],
    wordy::Bool = false,
    solvekwds...) where {T<:AbstractFloat,N}
                
    wordy && println("+ Estimation of star model")

    # operators
    gamma = lambda_ref(lambda) ./lambda
    rho_nu_star = rho .- nu_star[1]
    Ix = Interpolator(ker, lambda, length(x))
    Iy = Interpolator(ker, rho_nu_star, length(y))
    F_star = SparseInterpolator(kernel(Ix), lambda, nodes(Ix))
    H_star = SparseInterpolator(kernel(Iy), gamma .* rho_nu_star, nodes(Iy))
    SED_x = vcreate(dat)

    iter = 0
    res_loss = zeros(size(dat))
    loss = 0.0
    loss_past = 0.0
    while true
        # Build B_star operator
        B_star = build_B_star(gamma, F_star, x, H_star)
        while true
            # Update on-axis PSF
            alpha_prev = alpha
            solve!(y, B_star, dat, wgt, (1/alpha^degree(Ry))*Ry;
                   nonnegative=nonnegative, solvekwds...)
            # Update optimal scaling
            alpha = best_alpha(x, Rx, y, Ry) # Eq. 24
            if iter > 0 || abs(alpha - alpha_prev) ≤ alphatol*abs(alpha)
                break
            end
        end

        # Build A_star operator
        A_star = build_A_star(gamma, H_star, y, F_star)
        # Update star SED
        solve!(x, A_star, dat, wgt, (alpha^degree(Rx))*Rx;
               nonnegative=nonnegative, solvekwds...)
        # Update optimal scaling
        alpha = best_alpha(x, Rx, y, Ry)

        # stop criterions
        iter += 1
        residuals_star!(res_loss, dat, lambda, rho, nu_star, x, y)
        loss = vdot(wgt, res_loss, res_loss) + call(Rx, x) + call(Ry, y)
        if iter ≥ maxiter || (abs(loss - loss_past) <= max(tol[1], tol[2]*abs(loss_past)))
            break
        end

        # auto-calibration of center of speckles compared to center of corono
        if auto_calib
            copyto!(SED_x, F_star*x)
            fit_nu_star!(nu_star, dat, wgt, gamma, rho, SED_x, y; nu_box=nu_box)
            copyto!(rho_nu_star, rho .- nu_star[1])
            Iy = Interpolator(ker, rho_nu_star, length(y))
            H_star = SparseInterpolator(kernel(Iy), gamma .* rho_nu_star, 
                                        nodes(Iy))
        end
        loss_past = loss
    end
    vscale!(x, alpha)
    vscale!(y, 1/alpha)

    return x, y, nu_star
end

function fitStar(x0::AbstractVector{T}, # stellar SED
    y0::AbstractVector{T}, # stellar on-axis PSF
    nu_star0::AbstractVector{T},
    D::CalibratedLSSData{T,N},
    mu_x::Real;
    kwds...) where {T<:AbstractFloat,N}
    
    Rx = mu_x*quadraticsmoothness
    Ry = quadraticsmoothness

    return fitStar!(copy(x0), copy(y0), copy(nu_star0), D.d, D.w, D.rho, 
                    D.lambda, Rx, Ry; kwds...)
end




"""
"""
function fitCompanion!(z::AbstractVector{T},
    nu_comp::AbstractVector{T},
    dat::AbstractArray{T,N}, # residual data Eq. (29e)
    wgt::AbstractArray{T,N},
    rho::AbstractArray{T,N},
    lambda::AbstractArray{T,N},
    nu_star::AbstractVector{T},
    Rz::HomogeneousRegularization;
    auto_calib::Bool = false,
    nonnegative::Bool = false,
    maxiter::Integer = 10,
    tol::Tuple{Real,Real} = (0.0, 1e-6),
    nu_box::AbstractVector{T} = [50.0, 2.0],
    wordy::Bool = false,
    solvekwds...) where {T,N}
    
    wordy && println("+ Estimation of companion model")

    # operators
    gamma = lambda_ref(lambda) ./ lambda
    rho_nu_star = rho .- nu_star[1]
    Iz = Interpolator(ker, lambda, length(z))
    F_comp = SparseInterpolator(kernel(Iz), lambda, nodes(Iz))
    H_comp = off_axis_PSF(lambda, rho_nu_star, nu_comp)
    SED_z = vcreate(dat)

    iter = 0
    res_loss = zeros(size(dat))
    loss = 0.0
    loss_past = 0.0
    while true
        # Build A_comp operator
        A_comp = build_A_comp(gamma, H_comp, F_comp)
        # Update companion SED
        solve!(z, A_comp, dat, wgt, Rz; nonnegative=nonnegative, solvekwds...)

        # stop criterions
        iter += 1
        residuals_comp!(res_loss, dat, lambda, rho, nu_star, nu_comp, z)
        loss = vdot(wgt, res_loss, res_loss) + call(Rz, z) 
        if iter ≥ maxiter || (abs(loss - loss_past) <= max(tol[1], tol[2]*abs(loss_past)))
            break
        end

        # auto-calibration of off-axis PSF
        if auto_calib
            copyto!(SED_z, F_comp*z)
            fit_nu_z!(nu_comp, dat, wgt, lambda, rho_nu_star, SED_z; nu_box=nu_box)
            H_comp = off_axis_PSF(lambda, rho_nu_star, nu_comp)
        end
        loss_past = loss
    end

    return z, nu_comp
end

function fitCompanion(z0::AbstractVector{T},
    nu_comp0::AbstractVector{T},
    D::CalibratedLSSData{T,N},
    nu_star::AbstractVector{T},
    mu_z::Real;
    kwds...) where {T<:AbstractFloat,N}
    
    Rz = mu_z*quadraticsmoothness
    
    return fitCompanion!(copy(z0), copy(nu_comp0), D.d, D.w, D.rho, 
                         D.lambda, nu_star, Rz; kwds...)
end

