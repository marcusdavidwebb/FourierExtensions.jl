struct FourierExtension2{T}
    Ω   # indicator function of Ω ⊂ [0,1] × [0,1]
    coeffs :: Matrix{T}
end

function FourierExtension2(f, Ω, n; tol = 1e-8, oversamp = 2.0)
    N = (2n[1]+1)*(2n[2]+1)
    L = ceil.(Int, 2oversamp.*n)
    x_grid, y_grid, gridrefs = grid_mask(L, Ω)
    while length(gridrefs) < oversamp*N
        L .*= 2
        x_grid, y_grid, gridrefs = grid_mask(L, Ω)
    end
    b = complex(f.(x_grid,y_grid')[gridrefs]/sqrt(prod(L)))
    M = length(b)
    padded_data = Matrix{eltype(b)}(undef, L)
    ifftplan! = plan_ifft!(padded_data)
    fftplan! = plan_fft!(padded_data)
    A = LinearMap(
        (output,x) -> fourier_ext_2D_A!(output, x, n, gridrefs, L, ifftplan!, padded_data),
        (output,y) -> fourier_ext_2D_Astar!(output, y, n, gridrefs, L, fftplan!, padded_data),
        M, N; ismutating=true)
    rank_guess = min(round(Int, 5*sqrt(N)*log10(N))+10, div(N,2))
    coeffs = AZ_algorithm(A, A, b; rank_guess, tol) # Z = A for Fourier extensions
    FourierExtension2(Ω, reshape(coeffs, 2n[1]+1, 2n[2]+1)), A, b
end

function fourier_ext_2D_A!(output, coef, n, gridrefs, L, ifftplan!, padded_data)
    nx, ny = n
    Lx, Ly = L
    c = reshape(coef, 2nx+1, 2ny+1)
    padded_data .= 0
    @views padded_data[1:nx+1,1:ny+1] = c[nx+1:2nx+1,ny+1:2ny+1]
    @views padded_data[1:nx+1,Ly-ny+1:Ly] = c[nx+1:2nx+1,1:ny]
    @views padded_data[Lx-nx+1:Lx,1:ny+1] = c[1:nx,ny+1:2ny+1]
    @views padded_data[Lx-nx+1:Lx,Ly-ny+1:Ly] = c[1:nx,1:ny]
    ifftplan!*padded_data
    @views output .= padded_data[gridrefs].*sqrt(prod(L))
    output
end

function fourier_ext_2D_Astar!(output, v, n, gridrefs, L, fftplan!, padded_data)
    nx, ny = n
    Lx, Ly = L
    padded_data .= 0
    @views padded_data[gridrefs] .= v./sqrt(prod(L))
    fftplan!*padded_data
    d = reshape(output, 2nx+1, 2ny+1)
    @views d[1:nx,1:ny] = padded_data[Lx-nx+1:Lx,Ly-ny+1:Ly]
    @views d[1:nx,ny+1:2ny+1] = padded_data[Lx-nx+1:Lx,1:ny+1]
    @views d[nx+1:2nx+1,1:ny] = padded_data[1:nx+1,Ly-ny+1:Ly]
    @views d[nx+1:2nx+1,ny+1:2ny+1] = padded_data[1:nx+1,1:ny+1]
    output
end

function grid_mask(L, Ω)
    x_grid = (0:L[1]-1)/L[1]
    y_grid = (0:L[2]-1)/L[2]
    Z = Ω.(x_grid,y_grid')
    gridrefs = findall(Z)
    x_grid, y_grid, gridrefs
end

function (F::FourierExtension2)(x,y)
    nx, ny = div.(size(F.coeffs),2)
    real(sum(F.coeffs[k+nx+1,j+ny+1] * exp(2π*im*(k*x + j*y)) for k = -nx:nx, j =-ny:ny))
end

function grid_eval(F::FourierExtension2, L)
    # Evaluates a Fourier extension F.Ω ⊂ [0,1]x[0,1]
    # with coefficients F.coeffs at the grid points F.Ω ∩ ((0:L)/L)×((0:L)/L)
    # vals_l = sum_{k,j=-n:n} c(k,j) exp(2pi*i*(k*x_l + j*y_l))
    nx, ny = div.(size(F.coeffs),2)
    Lx, Ly = L
    @assert (Lx ≥ 2nx+1) & (Ly ≥ 2ny+1)
    c = F.coeffs
    padded_data = zeros(eltype(c),L)
    @views padded_data[1:nx+1,1:ny+1] = c[nx+1:2nx+1,ny+1:2ny+1]
    @views padded_data[1:nx+1,Ly-ny+1:Ly] = c[nx+1:2nx+1,1:ny]
    @views padded_data[Lx-nx+1:Lx,1:ny+1] = c[1:nx,ny+1:2ny+1]
    @views padded_data[Lx-nx+1:Lx,Ly-ny+1:Ly] = c[1:nx,1:ny]
    ifft!(padded_data)
    x_grid, y_grid, gridrefs = grid_mask(L, F.Ω)
    vals = real.(padded_data[gridrefs].*sqrt(prod(L)))
    x_grid, y_grid, gridrefs, vals
end

function Plots.contourf(F::FourierExtension2, L)
    x_grid, y_grid, gridrefs, vals = grid_eval(F, L)
    valsmasked = Matrix{eltype(vals)}(undef,L)*NaN
    @views valsmasked[gridrefs] .= vals
    contourf(x_grid, y_grid, valsmasked', aspect_ratio=1, xlabel="x", ylabel = "y")
end