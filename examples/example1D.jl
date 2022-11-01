using FourierExtensions
f = FourierExtension(x->cos(x^2)+x^7 + 1)
using Plots
plot(f)