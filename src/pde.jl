export derivative,
    pde_matrix,
    differentiation_matrix

"Compute the differentiation matrix of a 1D Fourier extension, with the given order."
differentiation_matrix(n::Int, order::Int) =
    Diagonal([(π*1im*k/2)^order for k in -n:n])

"Compute the differentiation matrix of a 2D Fourier extension, with the given orders."
function differentiation_matrix(n::Tuple{Int,Int}, order::Tuple{Int,Int})
    nx, ny = n
    d = [(2π*1im*k)^order[1]*(2π*1im*j)^order[2] for k in -nx:nx, j in -ny:ny]
    Diagonal(d[:])
end

"Return the derivative of a Fourier extension."
function derivative(f::FourierExtension, order::Int = 1)
    n = (length(f.coeffs)-1) >> 1
    D = differentiation_matrix(n, order)
    FourierExtension(D*f.coeffs)
end

function derivative(f::FourierExtension2, order::Tuple{Int,Int})
    n = (size(f.coeffs) .-1) .>> 1
    D = differentiation_matrix(n, order)
    FourierExtension2(f.Ω, reshape(D*f.coeffs[:], 2 .*n .+ 1))
end

function padded_mv!(output, x, A::LinearMap, E, K)
    M2 = length(output)
    M = M2-K
    output[1:M] = A * x
    output[M+1:M+K] = E * x
    output
end

function padded_mv_adj!(output, y, A::LinearMap, E, K)
    N = length(output)
    M = size(A,1)
    output[:] = A' * y[1:M]
    output[:] = output[:] + E'*y[M+1:M+K]
    output
end

function fe_1d_setup(n::Int; oversamp=2, T = Float64)
    m = ceil(Int, oversamp*n)
    grid = (-1:one(T)/m:1)
    padded_data = Vector{Complex{T}}(undef,4m)
    ifftplan! = plan_bfft!(padded_data)
    fftplan! = plan_fft!(padded_data)
    A = LinearMap(
        (output,x) -> fourier_ext_A!(output,x,n,m,ifftplan!,padded_data),
        (output,y) -> fourier_ext_Astar!(output,y,n,m,fftplan!,padded_data),
        2m+1, 2n+1; ismutating=true)
    A, m, grid
end

evalmatrix_1d_basis(n, x) = [exp(π*im*j*x/2) for j in -n:n]

function solve_constant_coefficient_ode(DiffOp::AbstractVector, f, bc::Tuple{T,T}, n::Int;
            tol = 1e-12, oversamp=2.0) where {T}
    A_map, m, grid = fe_1d_setup(n; oversamp, T)
    M, N = 2m+1, 2n+1
    D = differentiation_matrix(n, 1)
    PD = sum(a*D^(k-1) for (k,a) in enumerate(DiffOp))
    PDinv = pinv(PD)
    A_dense = Matrix{Complex{T}}(A_map)
    K = 2       # number of boundary conditions

    ## Direct solve
    A1 = zeros(Complex{T}, M+K, N)
    A1[1:M,1:N] = A_dense * PD
    A_bc = zeros(Complex{T}, K, N)
    A_bc[1,1:N] = evalmatrix_1d_basis(n, -one(T))
    A_bc[2,1:N] = evalmatrix_1d_basis(n, one(T))
    A1[M+1:M+K,1:N] = A_bc

    b = zeros(Complex{T}, M+K)
    b[1:M] = complex(f.(grid))
    b[M+1] = bc[1]
    b[M+2] = bc[2]

    @time c1 = A1 \ b
    F1 = FourierExtension(c1)

    ## AZ algorithm with dense matrices: option 1
    A2_dense = zeros(Complex{T}, M+K, N)
    A2_dense[1:M,1:N] = A_dense * PD * PDinv
    A_bc = zeros(Complex{T}, K, N)
    A_bc[1,1:N] = evalmatrix_1d_basis(n, -one(T))
    A_bc[2,1:N] = evalmatrix_1d_basis(n, one(T))
    A2_dense[M+1:M+K,1:N] = A_bc * PDinv

    BC0 = zeros(Complex{T}, K, N)
    Z2_dense = [A_dense/4m; BC0]

    A2_map = LinearMap(
        (output,x) -> padded_mv!(output, x, A_map*(PD*PDinv), A_bc * PDinv, K),
        (output,y) -> padded_mv_adj!(output, y, A_map*(PD*PDinv), A_bc * PDinv, K),
        M+K, N; ismutating=true
    )
    Z2_map = LinearMap(
        (output,x) -> padded_mv!(output, x, A_map/4m, BC0, K),
        (output,y) -> padded_mv_adj!(output, y, A_map/4m, BC0, K),
        M+K, N; ismutating=true
    )

    rank_guess = min(ceil(Int, 8*log(2n+1))+10, 2n+1)
    # d2 = AZ_algorithm(A2_dense, Z2_dense, b; rank_guess, tol)
    @time d2 = AZ_algorithm(A2_map, Z2_map, b; rank_guess, tol)
    c2 = PDinv * d2
    F2 = FourierExtension(c2)

    ## AZ algorithm with dense matrices: option 2
    A3_dense = zeros(Complex{T}, M+K, N)
    A3_dense[1:M,1:N] = A_dense * PD
    A3_dense[M+1:M+K,1:N] = A_bc
    Z3_dense = [A_dense*PDinv'/4m; BC0]

    A3_map = LinearMap(
        (output,x) -> padded_mv!(output, x, A_map*PD, A_bc, K),
        (output,y) -> padded_mv_adj!(output, y, A_map*PD, A_bc, K),
        M+K, N; ismutating=true
    )
    Z3_map = LinearMap(
        (output,x) -> padded_mv!(output, x, A_map*PDinv'/4m, BC0, K),
        (output,y) -> padded_mv_adj!(output, y, A_map*PDinv'/4m, BC0, K),
        M+K, N; ismutating=true
    )

    rank_guess = min(ceil(Int, 8*log(2n+1))+10, 2n+1)
    # c3 = AZ_algorithm(A3_dense, Z3_dense, b; rank_guess, tol)
    @time c3 = AZ_algorithm(A3_map, Z3_map, b; rank_guess, tol)
    F3 = FourierExtension(c3)

    F1, F2, F3
end

# This is a copy of the code in fourier_ext2D for now. Todo: clean up.
function fe_2d_setup(Ω, n, oversamp)
    N = (2n[1]+1)*(2n[2]+1)
    L = ceil.(Int, 2oversamp.*n)
    grid, gridΩrefs = grid_mask(Ω, L)
    while length(gridΩrefs) < oversamp*N # try to ensure oversampling rate
        L = L .* 2
        grid, gridΩrefs = grid_mask(Ω, L)
    end
    M = length(gridΩrefs)
    padded_data = Matrix{ComplexF64}(undef, L)
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
    L, M, N, Amap, grid, gridΩrefs = fe_2d_setup(Ω, n, oversamp)
    A_f = Matrix{ComplexF64}(Amap)
    Ap_f = Matrix{ComplexF64}(Amap')
    D = differentiation_matrix(n, (2,0)) + differentiation_matrix(n, (0,2))
    P = Diagonal(p.(grid[1], grid[2]')[gridΩrefs])
    b_f = complex(f.(grid[1], grid[2]')[gridΩrefs])
    bc_pts = sample_boundary(∂Ω, K)
    b_bc = [g(x[1],x[2]) for x in bc_pts]

    b = [b_f; b_bc]
    A = zeros(ComplexF64, M+K, N)
    A[1:M,1:N] = A_f * D + P * A_f
    for k in 1:K
        Z = eval_2d_basis(n, bc_pts[k][1], bc_pts[k][2])
        A[M+k,1:N] = Z[:]
    end
    c = A \ b
    coeffs = reshape(c, 2 .* n .+ 1)
    F = FourierExtension2(Ω, coeffs)
    F, A, b
end
