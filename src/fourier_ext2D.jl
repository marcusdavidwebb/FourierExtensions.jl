struct FourierExtension2{T}
    Ω   # indicator function of Ω ⊂ [0,1] × [0,1]
    coeffs :: Matrix{T}
end

function FourierExtension2(f, Ω, n::Tuple{Int,Int}; tol = 1e-12, oversamp = 2)
    L = ceil.(Int, 2oversamp.*n)
    grid, gridΩrefs = grid_mask(Ω, L)
    N = (2n[1]+1)*(2n[2]+1)
    while length(gridΩrefs) < oversamp * N # try to ensure oversampling rate
        L = L .* 2
        grid, gridΩrefs = grid_mask(Ω, L)
    end
    b = complex(f.(grid[1], grid[2]')[gridΩrefs])
    M = length(b)
    padded_data = Matrix{eltype(b)}(undef, L)
    ifftplan! = plan_bfft!(padded_data)
    fftplan! = plan_fft!(padded_data)
    A = LinearMap(
        (output,x) -> fourier_ext_2D_A!(output, x, n, gridΩrefs, ifftplan!, padded_data),
        (output,y) -> fourier_ext_2D_Astar!(output, y, n, gridΩrefs, fftplan!, padded_data),
        M, N; ismutating=true)
    rank_guess = min(round(Int, 4*sqrt(N)*log10(N))+20, div(N,2))
    coeffs = AZ_algorithm(A, A, b; rank_guess, tol) # Z = A for Fourier extensions
    FourierExtension2(Ω, reshape(coeffs, 2n[1]+1, 2n[2]+1))
end

function fourier_ext_2D_A!(output, coeffs, n::Tuple{Int,Int}, gridΩrefs, ifftplan!, padded_data)
    nx, ny = n
    Lx, Ly = size(padded_data)
    c = reshape(coeffs, 2nx+1, 2ny+1)
    padded_data .= 0
    @views padded_data[1:nx+1,1:ny+1] = c[nx+1:2nx+1,ny+1:2ny+1]
    @views padded_data[1:nx+1,Ly-ny+1:Ly] = c[nx+1:2nx+1,1:ny]
    @views padded_data[Lx-nx+1:Lx,1:ny+1] = c[1:nx,ny+1:2ny+1]
    @views padded_data[Lx-nx+1:Lx,Ly-ny+1:Ly] = c[1:nx,1:ny]
    ifftplan!*padded_data
    @views output .= padded_data[gridΩrefs]
    output
end

function fourier_ext_2D_Astar!(output, vals, n::Tuple{Int,Int}, gridΩrefs, fftplan!, padded_data)
    nx, ny = n
    Lx, Ly = size(padded_data)
    padded_data .= 0
    @views padded_data[gridΩrefs] .= vals./(Lx*Ly)
    fftplan!*padded_data
    d = reshape(output, 2nx+1, 2ny+1)
    @views d[1:nx,1:ny] = padded_data[Lx-nx+1:Lx,Ly-ny+1:Ly]
    @views d[1:nx,ny+1:2ny+1] = padded_data[Lx-nx+1:Lx,1:ny+1]
    @views d[nx+1:2nx+1,1:ny] = padded_data[1:nx+1,Ly-ny+1:Ly]
    @views d[nx+1:2nx+1,ny+1:2ny+1] = padded_data[1:nx+1,1:ny+1]
    output
end

function grid_mask(Ω, L::Tuple{Int,Int})
    grid = (0:L[1]-1)/L[1], (0:L[2]-1)/L[2]
    gridΩrefs = findall(Ω.(grid[1],grid[2]')) # find all grid pts inside Ω
    grid, gridΩrefs
end

# Evaluates a 2D Fourier extension at (x,y)
function (F::FourierExtension2)(x,y)
    nx, ny = div.(size(F.coeffs),2)
    real(sum(F.coeffs[k+nx+1,j+ny+1] * exp(2π*im*(k*x + j*y)) for k = -nx:nx, j =-ny:ny))
end

# Evaluates a 2D Fourier extension at the grid points F.Ω ∩ ((0:L1)/L1)×((0:L2)/L2)
function grid_eval(F::FourierExtension2, L::Tuple{Int,Int})
    nx, ny = div.(size(F.coeffs),2)
    Lx, Ly = L
    @assert (Lx ≥ 2nx+1) & (Ly ≥ 2ny+1)
    padded_data = zeros(eltype(F.coeffs),L)
    @views padded_data[1:nx+1,1:ny+1] = F.coeffs[nx+1:2nx+1,ny+1:2ny+1]
    @views padded_data[1:nx+1,Ly-ny+1:Ly] = F.coeffs[nx+1:2nx+1,1:ny]
    @views padded_data[Lx-nx+1:Lx,1:ny+1] = F.coeffs[1:nx,ny+1:2ny+1]
    @views padded_data[Lx-nx+1:Lx,Ly-ny+1:Ly] = F.coeffs[1:nx,1:ny]
    bfft!(padded_data)
    grid, gridΩrefs = grid_mask(F.Ω, L)
    vals = real(padded_data[gridΩrefs])
    grid, gridΩrefs, vals
end

function Plots.contourf(F::FourierExtension2, L::Tuple{Int,Int})
    grid, gridΩrefs, vals = grid_eval(F, L)
    valsmasked = fill(eltype(vals)(NaN),L)
    @views valsmasked[gridΩrefs] .= vals
    contourf(grid[1], grid[2], valsmasked', aspect_ratio=1, xlabel="x", ylabel = "y")
end
