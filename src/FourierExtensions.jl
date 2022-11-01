module FourierExtensions

using LinearMaps
using LinearAlgebra
using FFTW
using Plots

export FourierExtension

include("fourier_ext1D.jl")
include("fourier_ext2D.jl")
include("AZ_algorithm.jl")

end # module FourierExtensions
