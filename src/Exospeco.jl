#
# Exospeco.jl --
#
# Exospeco is a module to extract the companion SED from pre-processed and 
# calibrated long-slit spectroscopy data.
#
#-------------------------------------------------------------------------------
#

module Exospeco


export CalibratedLSSData,
       Interpolator,
       exospeco!,
       exospeco,
       fitCompanion!,
       fitCompanion,
       fitStar!,
       fitStar,
       get_data,
       get_weights,
       get_spatial_law,
       get_spectral_law,
       initialize!,
       initialize,
       lambda_ref,
       quadraticsmoothness,
       residuals!,
       residuals,
       residuals_comp!,
       residuals_comp,
       residuals_star!,
       residuals_star
       
       
import Base: show
using InterpolationKernels
using LinearAlgebra
using LazyAlgebra
using LinearInterpolators
using OptimPackNextGen
import OptimPackNextGen.BraDi
import OptimPackNextGen.Brent
import OptimPackNextGen.Powell.Bobyqa
import OptimPackNextGen.Powell.Newuoa


""" Wavelengh units (all wavelengths in nanometers) """
const nm  = 1.0    # one nanometer
const µm  = 1000nm # one micrometer

""" Angular distance units (all angular distances in milliarcseconds) """
const mas = 1.0    # one milliarcsecond

""" Angular distance by pixel """
const rho_pixel = 12.27mas

""" Interpolation kernel """
const ker = CatmullRomSpline(Float64)

include("types.jl")
include("model.jl")
include("initialization.jl")
include("algorithms.jl")
include("estimation_tools.jl")

end # module
