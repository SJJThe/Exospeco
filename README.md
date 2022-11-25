# Extraction Of SPEctrum of COmpanion in long-slit spectroscopy data

`Exospeco` is a [Julia](https://julialang.org/) package used in high-contrast imaging for extracting the sub-stellar companion SED in long-slit spectroscopy data.

## Installation

To install the package, hit the `]` key to activate the `Pkg` mode of julia and enter the following line:

```julia
add https://github.com/SJJThe/Exospeco
```


## Usage

### Geometric calibration

The first step toward using `Exospeco` is to obtain the geometric calibration. Either load pre-existing angular separations map `rho`, wavelengths map `lambda` and mask of valid data `mask` into a `GeoCalib` structure:

```julia
using ExospecoCalibration

Geo = GeoCalib(rho, lambda, mask)
```


or use the [ExospecoCalibration](https://github.com/SJJThe/ExospecoCalibration) to generate these maps from a calibration data `d_cal` and a bad pixels map `bpm`:

```julia
using ExospecoCalibration

pol_rho = (2,1)
pol_lambda = (2,5)
Geo = calibrate_geometry(d_cal, bpm; spatial_law_carac=pol_rho, spectral_law_carac=pol_lambda)
```


If a geometric calibration already exists, it is possible to load it by:

```julia
path_to_struct = "..."
Geo = readfits(GeoCalib, path_to_struct)
```


Thanks to this calibration, it is possible to build a `CalibratedLSSData` structure given a frame `d` of LSS data and their respective weights `w`:

```julia
using Exospeco

D = CalibratedLSSData(d, w, Geo)
```


### Initialization

The next step is to initialize the parameters of the problem that will be estimated by `Exospeco`, that is `x` the star SED, `y` the on-axis PSF, `nu_star` the angular separation between the star and the center of the coronagraphic mask, `z` the companion SED and `nu_comp` the parameters of the off-axis PSF:

```julia
# number of samples of x, y, z
N_x = 1024
N_y = 1024
N_z = 1024
# angular separation from companion to center of coronagraphic mask
rho_comp = -800*Exospeco.mas

(x_init, y_init, nu_star_init, z_init,
 nu_comp_init) = initialize(N_x, N_y, N_z, D, rho_comp; wordy=true)
```


### Estimation of the parameters

`Exospeco` needs the values of the hyper-parameters `mu_x` and `mu_z` corresponding to the levels of the star SED regularization and the companion SED regularization:

```julia
mu_x = 1e1
mu_z = 5e4

(x, y, nu_star, z,
 nu_comp) = exospeco(x_init, y_init, nu_star_init, z_init, nu_comp_init, D, 
                     mu_x, mu_z; delay_auto_calib=true, 
                     nonnegative=true, maxiter=20, alg_tol=(0.0,1e-3), 
                     wordy=true)
```

