
using EasyFITS
using Exospeco
using ExospecoCalibration
using PyPlot
const plt = PyPlot


# Load data
path_to_data = joinpath(@__DIR__, "..", "data")

d = read(FitsArray, joinpath(path_to_data, "preprocessed_data.fits.gz"))
w = read(FitsArray, joinpath(path_to_data, "preprocessed_data_weights.fits.gz"))


# Load or generate geometric calibration
Geo = readfits(GeoCalib, joinpath(path_to_data, "geometric_calibration.fits.gz"))


# Form calibrated LSS data used by Exospeco
D = CalibratedLSSData(d, w, Geo)

figure()
plt.imshow(D.d, origin="lower", interpolation="none", aspect="auto", 
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
solvekwds = (ftol=(0.0,1e-5), verb=true)

(x, y, nu_star, z,
 nu_comp) = exospeco(x_init, y_init, nu_star_init, z_init, nu_comp_init, D, 
                     mu_x, mu_z; auto_calib=true, delay_auto_calib=true, 
                     nonnegative=true, maxiter=20, alg_tol=(0.0,1e-3), 
                     wordy=true, solvekwds...)

res = residuals(D.d, D.lambda, D.rho, nu_star, nu_comp, y, x, z)

figure()
plt.imshow(res, origin="lower", cmap="seismic", 
           interpolation="none", aspect="auto", vmin=-0.15, vmax=0.15)
plt.colorbar()
plt.tight_layout()
    
Iz = Interpolator(Exospeco.ker, D.lambda, length(z))
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

