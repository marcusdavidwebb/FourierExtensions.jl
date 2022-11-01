using FourierExtensions
f = (x,y) -> x^2 + y^2 + x - cos(y)
Ω = (x,y) -> (x-.5)^2 + (y-.5)^2 < 0.2
n = 12
F, A, b = FourierExtension2(f,Ω, n)
using LinearAlgebra
norm(A*F.coeffs[:] - b)

b
AA = Matrix{ComplexF64}(A) 
c = AA\b
norm(AA*c - b)

AA' * v
A' * v

AAs = Matrix{ComplexF64}(A')'
AAs' 
norm(AA-AAs)
AA[3,4]/conj(AAs'[4,3])
c = AA\b
norm(AA*c - b)

c = reshape(c,2n+1,2n+1)
G = FourierExtension2{ComplexF64}(Ω,c)
F(0.5,0.55)
N = (2n+1)^2
G(0.5,0.55)
f(0.5,0.55)
f(0.5,0.55)/G(0.5,0.55)




n = 5
oversamp = 2
N = (2n+1)^2
L = ceil(Int, 4*oversamp*n)
xgrid = 0:1/L:1-1/L
ygrid = copy(xgrid)
indsx, indsy = FourierExtensions.grid_filter(xgrid, ygrid, Ω)
M = length(indsx)
while M < oversamp*N
    L *= 2
    xgrid = 0:1/L:1-1/L
    ygrid = copy(xgrid)
    indsx, indsy = grid_filter(xgrid, ygrid, Ω)
    M = length(indsx)
end

b = complex([f(xgrid[indsx[k]],ygrid[indsy[k]]) for k = 1:M]) / L
rank_guess = min(round(Int, 5*sqrt(N)*log10(N))+10, div(N,2))
padded_tensor = Array{eltype(b),3}(undef, L, L, rank_guess)
bfftplan = FourierExtensions.plan_bfft!(padded_tensor[:,:,1])
using LinearMaps
import FourierExtensions: fourier_ext_2D_A!, fourier_ext_2D_Astar!
A = LinearMap(x -> fourier_ext_2D_A!(x,n,indsx,indsy,L,bfftplan,padded_tensor), y -> fourier_ext_2D_Astar!(y,n,indsx,indsy,L,bfftplan,padded_tensor), M, N)
Am = Matrix(A)
c = randn(N,rank_guess)
r = size(c,2)
c = reshape(c, 2n+1, 2n+1, r)
T = ComplexF64
padded_tensor[:,:,1:r] = [c[n+1:2n+1,n+1:2n+1,:] zeros(T,n+1,L-(2n+1),r) c[n+1:2n+1,1:n,:];
                                                    zeros(T,L-(2n+1),L,r);
                            c[1:n,n+1:2n+1,:]      zeros(T,n,L-(2n+1),r)   c[1:n,1:n,:]]
print("A norm: ")
print(norm(padded_tensor[:,:,1:r]))
println()
for j = 1:r
    bfftplan*view(padded_tensor,:,:,j)
end
vals = [padded_tensor[indsx[k],indsy[k],j] for k=1:length(indsx), j=1:r]
norm(vals)



n = 5
k,j = 2,6
c = zeros(ComplexF64,2n+1,2n+1)
c[k,j] = 1
Ω = (x,y) -> (x-.5)^2 + (y-.5)^2 < 0.2
F = FourierExtension2(Ω,c)
f = (x,y) -> exp(2pi*im*(((-n+k-1)*x +(-n+j-1)*y)))

F(0.1,0.2)
f(0.1,0.2)
F, A, b = FourierExtension2(f,Ω, n)
F(0.1,0.2)

cnew = Matrix{ComplexF64}(A)\b
cnew = reshape(cnew,2n+1,2n+1)
cnew[k,j]