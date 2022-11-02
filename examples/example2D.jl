using FourierExtensions
f = (x,y) -> exp(10*x) + y^2 + x - cos(y)
Ω = (x,y) -> (x-.5)^2 + (y-.5)^2 < 0.2
n = (12,15)
@time F, A, b = FourierExtension2(f,Ω, n);
using Plots
@time contourf(F,(100,120))
