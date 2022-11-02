struct FourierExtension2{T}
    Ω   # indicator function of Ω ⊂ [0,1] × [0,1]
    coeffs :: Matrix{T}
end

function FourierExtension2(f, Ω, n::Int; tol = 1e-8, oversamp = 2.0)
    N = (2n+1)^2
    L = ceil(Int, 4*oversamp*n)
    xgrid = 0:1/L:1-1/L
    ygrid = copy(xgrid)
    indsx, indsy = grid_filter(xgrid, ygrid, Ω)
    # grid, gridrefs = grid_mask(L, L, Ω)
    # M = length(gridrefs)

    M = length(indsx)
    while M < oversamp*N
        L *= 2
        xgrid = 0:1/L:1-1/L
        ygrid = copy(xgrid)
        indsx, indsy = grid_filter(xgrid, ygrid, Ω)
        M = length(indsx)
    end
    b = complex([f(xgrid[indsx[k]],ygrid[indsy[k]]) for k = 1:M])/L
    rank_guess = min(round(Int, 5*sqrt(N)*log10(N))+10, div(N,2))
    padded_data = Matrix{eltype(b)}(undef, L, L)
    ifftplan! = plan_ifft!(padded_data)
    fftplan! = plan_fft!(padded_data)
    A = LinearMap(
        (output,x) -> fourier_ext_2D_A!(output, x, n, indsx, indsy, L, ifftplan!, padded_data),
        (output,y) -> fourier_ext_2D_Astar!(output, y, n, indsx, indsy, L, fftplan!, padded_data),
        M, N; ismutating=true)
    coeffs = AZ_algorithm(A, A, b, rank_guess=rank_guess, tol=tol) # Z = A for Fourier extensions
    FourierExtension2(Ω, reshape(coeffs,2n+1,2n+1)), A, b
end

function fourier_ext_2D_A!(output, coef, n::Int, indsx::Vector{Int64}, indsy::Vector{Int64}, L::Int, ifftplan!, padded_data::AbstractArray)
    c = reshape(coef, 2n+1, 2n+1)
    padded_data .= 0
    @views padded_data[1:n+1,1:n+1] = c[n+1:2n+1,n+1:2n+1]
    @views padded_data[1:n+1,L-n+1:L] = c[n+1:2n+1,1:n]
    @views padded_data[L-n+1:L,1:n+1] = c[1:n,n+1:2n+1]
    @views padded_data[L-n+1:L,L-n+1:L] = c[1:n,1:n]
    ifftplan!*padded_data
    for k ∈ eachindex(indsx)
        output[k] = padded_data[indsx[k],indsy[k]]*L
    end
    # for k in 1:length(gridrefs)
    #     output[k] = padded_tensor[gridrefs[k]]/L
    # end
    output
end

function fourier_ext_2D_Astar!(output, v, n::Int, indsx::Vector{Int}, indsy::Vector{Int}, L::Int, fftplan!, padded_data::AbstractArray)
    padded_data .= 0
    for k ∈ eachindex(indsx)
        padded_data[indsx[k],indsy[k]] = v[k]/L
    end
    fftplan!*padded_data
    d = reshape(output, 2n+1, 2n+1)
    @views d[1:n,1:n] = padded_data[L-n+1:L,L-n+1:L]
    @views d[1:n,n+1:2n+1] = padded_data[L-n+1:L,1:n+1]
    @views d[n+1:2n+1,1:n] = padded_data[1:n+1,L-n+1:L]
    @views d[n+1:2n+1,n+1:2n+1] = padded_data[1:n+1,1:n+1]
    output
end

"""
Compute the subset of an `L1 × L2` regular grid consisting of points
that belong to a domain, defined by the characteristic function `Ω`.
"""
function grid_mask(L1, L2, Ω)
    x_grid = (0:L1-1)/L1
    y_grid = (0:L2-1)/L2
    grid = [(x,y) for x in x_grid, y in x_grid]
    Z = Ω.(grid)
    gridrefs = findall(Z)
    grid, gridrefs
end

function grid_filter(xgrid::AbstractVector,ygrid::AbstractVector,Ω)
    indsx = Vector{Int}(undef,0)
    indsy = copy(indsx)
    nx = length(xgrid)
    ny = length(ygrid)
    for jx = 1:nx, jy = 1:ny
            if Ω(xgrid[jx],ygrid[jy]) == 1
                push!(indsx, jx)
                push!(indsy, jy)
            end
    end
    indsx, indsy
end

function (F::FourierExtension2{T})(x,y) where T
    val = zero(T); nx, ny = div.(size(F.coeffs),2)
    for k = -nx:nx, j = -ny:ny
        val += F.coeffs[k+nx+1,j+ny+1]*exp(2π*im*(k*x + j*y))
    end
    val
end

function grid_eval(F::FourierExtension2{T}, L::Int) where T
    # Evaluates a Fourier extension F.Ω ⊂ [0,1]x[0,1]
    # with coefficients F.coeffs at the grid points F.Ω ∩ ((0:L)/L)×((0:L)/L)
    # vals_l = sum_{k,j=-n:n} c(k,j) exp(2pi*i*(k*x_l + j*y_l))
    n = div(size(F.coeffs,1),2)
    padded_mat = zeros(T,L,L)
    if L >= 2n+1
        @views padded_data[1:n+1,1:n+1] = F.coeffs[n+1:2n+1,n+1:2n+1]
        @views padded_data[1:n+1,L-n+1:L] = F.coeffs[n+1:2n+1,1:n]
        @views padded_data[L-n+1:L,1:n+1] = F.coeffs[1:n,n+1:2n+1]
        @views padded_data[L-n+1:L,L-n+1:L] = F.coeffs[1:n,1:n]
    else
        for k = 1:2n+1, j = 1:2n+1
                padded_mat[1+mod(k-1-n,L),1+mod(j-1-n,L)] = padded_mat[1+mod(k-1-n,L),1+mod(j-1-n,L)] + F.coeffs[k,j]
        end
    end
    bfft!(padded_mat)
    indsx, indsy = grid_filter(0:1/L:1L, 0:1/L:1, F.Ω)
    vals = [padded_mat[indsx[k],indsy[k]] for k ∈ eachindex(indsx)]
    indsx, indsy, vals
end


function Plots.contourf(F::FourierExtension2{T}, L) where T
    indsx,indsy,vals = grid_eval(F, L)
    M = length(indsx)
    if norm(imag(vals),Inf) < 1e-4
        valsmasked = Matrix{real(T)}(undef,L,L)*NaN
        for k = 1:M
            valsmasked[indsy[k],indsx[k]] = real(vals[k]);
        end
        contourf((0:L-1)/L,(0:L-1)/L, valsmasked, aspect_ratio=1, xlabel="x", ylabel = "y")
    else
        realvalsmasked = Matrix{real(T)}(undef,L,L)*NaN
        imagvalsmasked = copy(realvalsmasked)
        for k = 1:M
            realvalsmasked[indsy[k],indsx[k]] = real(vals[k])
            imagvalsmasked[indsy[k],indsx[k]] = imag(vals[k])
        end
        p1 = contourf((0:L-1)/L,(0:L-1)/L,realvalsmasked, title="Real Part", xlabel = "x", ylabel = "y", aspect_ratio=:equal)
        p2 = contourf((0:L-1)/L,(0:L-1)/L,imagvalsmasked, title="Imaginary Part", xlabel = "x", ylabel = "y", aspect_ratio=:equal)
        plot(p1,p2,layout=(1,2))
    end
end
