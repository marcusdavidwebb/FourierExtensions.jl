using FourierExtensions
f = (x,y) -> x^2 + y^2 + x - cos(y)
Ω = (x,y) -> (x-.5)^2 + (y-.5)^2 < 0.2
n = 12
@time F, A, b = FourierExtension2(f,Ω, n);
