using FourierExtensions
using LinearAlgebra, Plots

p = (x,y) -> cos(x+y)
f = (x,y) -> x^2 + y^2 + x - cos(y)
g = (x,y) -> sin(x-y)

function doughnut(x,y)
    r = sqrt((x-0.5)^2+(y-0.5)^2)
    0.1 ≤ r ≤ 0.3
end
function doughnut_boundary(t)
    @assert 0 ≤ t ≤ 1
    r1 = 0.1
    r2 = 0.3
    if t ≤ 0.5
        r1 .* (cos(4*pi*t), sin(4*pi*t))
    else
        r2 .* (cos(4*pi*(t-0.5)), sin(4*pi*(t-0.5)))
    end
end

n = (10,12)
F = FourierExtension2(f, doughnut, n)

Plots.contourf(F, (100,100))

K = 50
oversamp = 2.0
F, A, b = pde_matrix(doughnut, doughnut_boundary, n, oversamp, K, f, p, g)

x = 0.6
y = 0.3
G = derivative(F, (2,2))
G(x,y) + p(x,y) * F(x,y)
f(x,y)
