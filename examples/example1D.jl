using Revise
using FourierExtensions
f = x->cos(100x^2)+x^7 + 1
@time F = FourierExtension(f);
f(0.6) - F(0.6)


using Plots
plot(F)
