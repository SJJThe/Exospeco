
using Revise
using EasyFITS
using Exospeco
using ExospecoCalibration
using PyPlot
const plt = PyPlot


# Load data
path_to_data = "/home/samuel/Documents/Data/IRDIS_LSS/HIP43620"

d = read(FitsArray, joinpath(path_to_data, "preprocessed_data.fits"))
w = read(FitsArray, joinpath(path_to_data, "preprocessed_data_weights.fits"))

# Load or generate geometric calibration
#=
geo_cal = read(FitsArray, joinpath(path_to_data, "geometric_calibration.fits"))
rho_map, lambda_map, mask = geo_cal[:,:,1], geo_cal[:,:,2], geo_cal[:,:,3]
=#
d_cal = read(FitsArray, joinpath(path_to_data, "preprocessed_lamp.fits"))
bpm = read(FitsArray, joinpath(path_to_data, "bad_pixels_map.fits"))
spectral_law_carac = (2,5)
spatial_law_carac = (2,1)
rho_map, lambda_map = calibrate_geometry(d_cal, bpm;
                                         spectral_law_carac=spectral_law_carac,
                                         spatial_law_carac=spatial_law_carac,
                                         wordy=true, study=Val(:log))
lambda_bnds = (920.0, 1870.0) .* ExospecoCalibration.nm
rho_bnds = (-30.0, 30.0) .* ExospecoCalibration.rho_pixel
mask = select_region_of_interest(rho_map, lambda_map, bpm; 
                                 lambda_bnds=lambda_bnds, rho_bnds=rho_bnds)


# Form calibrated LSS data used by Exospeco
D = CalibratedLSSData(d, w .* mask, rho_map, lambda_map)

figure()
plt.imshow(get_data(D), origin="lower", interpolation="none", aspect="auto", 
           cmap="gnuplot", vmin=0.0, vmax=1.0)
plt.colorbar()


# Initialization
# number of samples of x, y, z
N_x = 1024
N_y = 1024
N_z = 1024
# angular separation from companion to center of coronagraphic mask
rho_comp = -800*Exospeco.mas

(x_init, y_init, nu_star_init, z_init,
 nu_comp_init) = initialize(N_x, N_y, N_z, D, rho_comp; wordy=true)


# Exospeco
mu_x = 1e1
mu_z = 5e4
solvekwds = (ftol=(0.0,1e-6), verb=true)

(x, y, nu_star, z,
 nu_comp) = exospeco(x_init, y_init, nu_star_init, z_init, nu_comp_init, D, 
                     mu_x, mu_z; auto_calib=true, delay_auto_calib=true, 
                     nonnegative=true, maxiter=20, alg_tol=(0.0,1e-3), 
                     wordy=true, solvekwds...)

res = residuals(get_data(D), get_spectral_law(D), get_spatial_law(D), nu_star, 
                nu_comp, y, x, z)

figure()
plt.imshow(res .* bpm, origin="lower", cmap="seismic", 
           interpolation="none", aspect="auto", vmin=-0.15, vmax=0.15)
plt.colorbar()
plt.tight_layout()
    
Iz = Interpolator(Exospeco.ker, lambda_map, length(z))
figure()
plt.plot(Exospeco.nodes(Iz), z)



#=

# Fit stellar parameters
mu_x = 1e1
solvekwds = (ftol=(0.0,1e-6), verb=true)

x, y, nu_star = fitStar(x_init, y_init, nu_star_init, D, mu_x; auto_calib=true, 
                        nonnegative=true, maxiter=20, tol=(0.0,1e-3), 
                        wordy=true, solvekwds...)


# Fit companion parameters
res_comp = residuals_comp(get_data(D), get_spectral_law(D), get_spatial_law(D),
                          nu_star, y, x)
D_comp = CalibratedLSSData(res_comp, get_weights(D), rho_map, lambda_map)
mu_z = 5e4

z, nu_comp = fitCompanion(z_init, nu_comp_init, D_comp, nu_star, mu_z;
                          auto_calib=true, nonnegative=true, maxiter=20, 
                          tol=(0.0,1e-3), wordy=true, solvekwds...)

Iz = Interpolator(Exospeco.ker, lambda_map, length(z))
figure()
plt.plot(Exospeco.nodes(Iz), z)

=#

