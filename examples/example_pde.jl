using FourierExtensions
using LinearAlgebra, Plots

###########
# an ODE
###########
f = x -> x^2 + cos(x)
g = x -> sin(x)+cos(x)

n = 120
T = Float64

wavenumber = 2.0
DiffOp = [wavenumber^2, 0, 1]  # 1D Helmholtz equation u'' + wavenumber^2*u

F1,F2,F3 = FourierExtensions.solve_constant_coefficient_ode(DiffOp, f, (g(-1),g(1)), n);

x0 = 0.4
@show F1(0.4)
@show F2(0.4)
@show F3(0.4)

@show abs(F1(-1)-g_1d(-1))
@show abs(F1(1)-g_1d(1))
G = derivative(F1, 2)
@show abs(G(x0) + wavenumber^2*F1(x0) - f_1d(x0))



#################################
# Variable coefficient PDE
#################################
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
Gxx = derivative(F, (2,0))
Gyy = derivative(F, (0,2))
@show Gxx(x,y) + Gyy(x,y) + p(x,y) * F(x,y)
@show f(x,y)


#################################
# Constant coefficient PDE
#################################

PdeCoeffs, PdeOrders = FourierExtensions.pde_Helmholtz(wavenumber=2.0)
f = (x,y) -> 0.0
g = (x,y) -> 1.0

K = 60
oversamp = 2.0
n = (20,25)
Ω = doughnut
∂Ω = doughnut_boundary

F = FourierExtensions.solve_constant_coefficient_pde(PdeCoeffs, PdeOrders, Ω, ∂Ω, f, g, n, oversamp, K)

x, y = 0.6, 0.3
x_bnd, y_bnd = ∂Ω(0.23)
Gxx = derivative(F, (2,0))
Gyy = derivative(F, (0,2))
@show Gxx(x,y) + Gyy(x,y) + wavenumber^2 * F(x,y) - f(x,y)
@show abs(F(x_bnd, y_bnd)- g(x_bnd,y_bnd))
