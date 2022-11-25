

function initialize!(x::AbstractVector{T},
    y::AbstractVector{T},
    nu_star::AbstractVector{T},
    z::AbstractVector{T},
    nu_comp::AbstractVector{T},
    dat::AbstractArray{T,N},
    wgt::AbstractArray{T,N},
    rho::AbstractArray{T,N},
    lambda::AbstractArray{T,N};
    wordy::Bool = false) where {T,N}
    
    wordy && println("+ Initialization of parameters")

    # init stellar SED x
    wordy && println("|- Init stellar SED x")
    lmin, lmax = extrema(lambda)
    lambda_star = range(lmin; length=length(x), stop=lmax)
    vfill!(x, T(0))
    for l in 1:length(lambda_star)
        lambdas_l = findmin(abs.(lambda .- lambda_star[l]), dims=2)[2]
        valid_pixels = wgt[lambdas_l] .!= 0.0
        nb_vp = length(dat[lambdas_l][valid_pixels])
        if nb_vp == 0
            x[l] = T(0)
        else
            x[l] = sum((wgt[lambdas_l] .* dat[lambdas_l])[valid_pixels]) /
            sum(wgt[lambdas_l])
        end
    end

    # init on-axis PSF y
    wordy && println("|- Init on-axis PSF y")
    scaled_rho = lambda_ref(lambda) ./lambda .* rho
    smin, smax = extrema(scaled_rho[:,end])
    scaled_rho_range = range(smin; length=length(y), stop=smax)
    vfill!(y, T(0))
    for s in 1:length(scaled_rho_range)
        rhos_s = findmin(abs.(scaled_rho .- scaled_rho_range[s]), dims=1)[2]
        valid_pixels = wgt[rhos_s] .!= 0.0
        nb_vp = length(dat[rhos_s][valid_pixels])
        if nb_vp == 0
            y[s] = T(0)
        else
            y[s] = sum((wgt[rhos_s] .* dat[rhos_s])[valid_pixels]) /
            sum(wgt[rhos_s])
        end
    end

    # init stellar geometric parameters nu_star
    wordy && println("|- Init stellar geometric parameters nu_star")
    nu_star = [0.0mas]
    wordy && println("|-- rho_star = ", nu_star[1], " mas from center of mask")

    # init companion SED z
    wordy && println("|- init companion SED z")
    vfill!(z, T(0))

    # init companion geometric parameters nu_comp
    wordy && println("|- init companion geometric parameters nu_comp")
    push!(nu_comp, 2rho_pixel)
    wordy && println("|-- rho_comp = ", nu_comp[1], " mas from center of mask")
    wordy && println("|-- sigma_comp = ", nu_comp[2], " mas of width")
    
    return x, y, nu_star, z, nu_comp
end

function initialize(N_x::Int,
    N_y::Int,
    N_z::Int,
    D::CalibratedLSSData{T,N},
    rho_comp::Real,
    rho_star::Real = 0.0;
    kwds...) where {T,N}

    return initialize!(zeros(N_x), zeros(N_y), [rho_star], zeros(N_z), [rho_comp], 
                       D.d, D.w, D.rho, D.lambda; kwds...)
end

