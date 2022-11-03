export derivative,
    pde_matrix

differentiation_matrix(n::Int, order::Int) =
    Diagonal([(π*1im*k/2)^order for k in -n:n])

function differentiation_matrix(n::Tuple{Int,Int}, order::Tuple{Int,Int})
    nx, ny = n
    d = [(2π*1im*k)^order[1]*(2π*1im*j)^order[2] for k in -nx:nx, j in -ny:ny]
    Diagonal(d[:])
end

function derivative(f::FourierExtension, order::Int = 1)
    n = (length(f.coeffs)-1) >> 1
    D = differentiation_matrix(n, order)
    FourierExtension(f.γ, D*f.coeffs)
end

function derivative(f::FourierExtension2, order::Tuple{Int,Int})
    n = (size(f.coeffs) .-1) .>> 1
    D = differentiation_matrix(n, order)
    FourierExtension2(f.Ω, reshape(D*f.coeffs[:], 2 .*n .+ 1))
end

# This is a copy of the code in fourier_ext2D for now. Todo: clean up.
function fe_2d_setup(Ω, n, oversamp, ::Type{T}) where {T}
    N = (2n[1]+1)*(2n[2]+1)
    L = ceil.(Int, 2oversamp.*n)
    grid, gridΩrefs = grid_mask(Ω, L)
    while length(gridΩrefs) < oversamp*N # try to ensure oversampling rate
        L = L .* 2
        grid, gridΩrefs = grid_mask(Ω, L)
    end
    M = length(gridΩrefs)
    padded_data = Matrix{Complex{T}}(undef, L)
    ifftplan! = plan_bfft!(padded_data)
    fftplan! = plan_fft!(padded_data)
    A = LinearMap(
        (output,x) -> fourier_ext_2D_A!(output, x, n, gridΩrefs, ifftplan!, padded_data),
        (output,y) -> fourier_ext_2D_Astar!(output, y, n, gridΩrefs, fftplan!, padded_data),
        M, N; ismutating=true)
    L, M, N, A, grid, gridΩrefs
end

function eval_2d_basis(n, x, y)
    nx, ny = n
    [exp(2π*im*(k*x + j*y)) for k in -nx:nx, j in -ny:ny]
end

"Sample K points on a boundary, using a parameterization on [0,1]."
sample_boundary(∂Ω, K) = ∂Ω.((0:K-1)/K)
"""
Solve the PDE Δu + p(x,y)u = f(x,y) on a domain Ω, subject to Dirichlet
boundary condition u = g(x,y) on ∂Ω.
"""
function pde_matrix(Ω, ∂Ω, n, oversamp, K, f, p, g)
    oversamp = 2.0
    T = Float64
    L, M, N, Amap, grid, gridΩrefs = fe_2d_setup(Ω, n, oversamp, T)
    A_f = Matrix{Complex{T}}(Amap)
    Ap_f = Matrix{Complex{T}}(Amap')
    D = differentiation_matrix(n, (2,2))
    P = Diagonal(p.(grid[1], grid[2]')[gridΩrefs])
    b_f = complex(f.(grid[1], grid[2]')[gridΩrefs])
    bc_pts = sample_boundary(∂Ω, K)
    b_bc = [g(x[1],x[2]) for x in bc_pts]

    b = [b_f; b_bc]
    A = zeros(Complex{T}, M+K, N)
    A[1:M,1:N] = A_f * D + P * A_f
    for k in 1:K
        Z = eval_2d_basis(n, bc_pts[k][1], bc_pts[k][2])
        A[M+k,1:N] = Z[:]
    end
    c = A \ b
    coeffs = reshape(c, 2 .* n .+ 1)
    F = FourierExtension2{Complex{T}}(Ω, coeffs)
    F, A, b
end
