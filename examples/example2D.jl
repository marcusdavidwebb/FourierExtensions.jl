using FourierExtensions
f = (x,y) -> x^2 + y^2 + x - cos(y)
Ω = (x,y) -> (x-.5)^2 + (y-.5)^2 < 0.2
F = FourierExtension2(f,Ω, 10)