using Revise, Plots
using FourierExtensions

f = (x,y) -> exp(x) + y^2 + x - cos(y)
Ω = (x,y) -> (x-.5)^2 + (y-.5)^2 < 0.2
n = (20,22)
@time F = FourierExtension2(f,Ω, n);
abs(F(0.51,0.48)-f(0.51,0.48))

contourf(F)


# Timings

# @time F = FourierExtension2(f,Ω, (6,6));
# @time F = FourierExtension2(f,Ω, (12,12));
# @time F = FourierExtension2(f,Ω, (24,24));
# @time F = FourierExtension2(f,Ω, (48,48));
# @time F = FourierExtension2(f,Ω, (96,96));

scatter([6^2;12^2;24^2;48^2;96^2],
  [0.007255;0.045697;0.480478;8.364671;128.381058],
  xscale=:log10, yscale=:log10, label="Timings",legend=:topleft)
  plot!([6^2;12^2;24^2;48^2;96^2], 1e-6*[6^2;12^2;24^2;48^2;96^2], label="O(N)")
  plot!([6^2;12^2;24^2;48^2;96^2], 1e-6*[6^2;12^2;24^2;48^2;96^2].^2, label="O(N^2)")
plot!([6^2;12^2;24^2;48^2;96^2], 1e-6*[6^2;12^2;24^2;48^2;96^2].^3, label="O(N^3)")
