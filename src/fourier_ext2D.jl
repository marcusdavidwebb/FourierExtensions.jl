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
    bfftplan = plan_bfft!(padded_data) # note that fft = L * conj ∘ bfft ∘ conj
    A = LinearMap(
        (output,x) -> fourier_ext_2D_A!(output, x, n, indsx, indsy, L, bfftplan, padded_data),
        (output,y) -> fourier_ext_2D_Astar!(output, y, n, indsx, indsy, L, bfftplan, padded_data),
        M, N; ismutating=true)
    coeffs = AZ_algorithm(A, A, b, rank_guess=rank_guess, tol=tol, maxiter=100) # Z = A for Fourier extensions
    FourierExtension2(Ω, reshape(coeffs,2n+1,2n+1)), A, b
end

# Adaptive Constructor
function FourierExtension2(f, Ω; tol=1e-8, oversamp = 2.0, nmin::Int=8, nmax::Int=32)
    n = nmin
    fval = f(0.51,0.49) # TODO: check at a point in Ω
    F = FourierExtension2(Ω,ones(complex(typeof(fval)),1,1))
    while n <= nmax
        F = FourierExtension2(f, Ω, n, oversamp=oversamp)
        indsx, indsy, vals = grid_evaluate(F, Ω, ceil(Int, 2*oversamp*n))
        fvals = f.(indsx,indsy)
        fvalsnorm = norm(fvals)
        if norm(F.coeffs)/fvalsnorm < 100 # check coefficients are not too large to be stable
            if norm(abs.(fvals - vals))/fvalsnorm < 10*tol # check residual
                if abs(F(0.51,0.49) - fval) < 10*tol # final check at a single "random" point
                    F
                end
            end
        end
        n *= 2
    end
    @warn "Maximum number of coefficients "*string(length(F.coeffs))*" reached in constructing FourierExtension2. Try setting nmax higher."
    F
end

function fourier_ext_2D_A!(output, coef, n::Int, indsx::Vector{Int64}, indsy::Vector{Int64}, L::Int, bfftplan, padded_data::AbstractArray)
    c = reshape(coef, 2n+1, 2n+1)
    @views padded_data[:,:] .= 0
    @views padded_data[1:n+1,1:n+1] = c[n+1:2n+1,n+1:2n+1]
    @views padded_data[1:n+1,L-n+1:L] = c[n+1:2n+1,1:n]
    @views padded_data[L-n+1:L,1:n+1] = c[1:n,n+1:2n+1]
    @views padded_data[L-n+1:L,L-n+1:L] = c[1:n,1:n]
    bfftplan*padded_data
    for k in 1:length(indsx)
        output[k] = padded_data[indsx[k],indsy[k]]/L
    end
    output
end

function fourier_ext_2D_Astar!(output, v, n::Int, indsx::Vector{Int}, indsy::Vector{Int}, L::Int, bfftplan, padded_data::AbstractArray)
    padded_data .= 0
    for k = 1:length(indsx)
        padded_data[indsx[k],indsy[k]] = conj(v[k])
    end
    bfftplan*padded_data
    d = reshape(output, 2n+1, 2n+1)
    @views d[1:n,1:n] = padded_data[L-n+1:L,L-n+1:L]
    @views d[1:n,n+1:2n+1] = padded_data[L-n+1:L,1:n+1]
    @views d[n+1:2n+1,1:n] = padded_data[1:n+1,L-n+1:L]
    @views d[n+1:2n+1,n+1:2n+1] = padded_data[1:n+1,1:n+1]
    d .= conj.(d)/L
    output
end

function grid_filter(xgrid::AbstractVector,ygrid::AbstractVector,Ω)
    indsx = Vector{Int}(undef,0)
    indsy = copy(indsx)
    nx = length(xgrid)
    ny = length(ygrid)
    for jx = 1:nx
        for jy = 1:ny
            if Ω(xgrid[jx],ygrid[jy]) == 1
                push!(indsx, jx)
                push!(indsy, jy)
            end
        end
    end
    indsx, indsy
end

function evaluate(F::FourierExtension2{T}, x, y) where T
    val = zero(T)
    nx, ny = div.(size(F.coeffs),2)
    for k = -nx:nx, j = -ny:ny
        val += F.coeffs[k+nx+1,j+ny+1]*exp(2π*im*(k*x + j*y))
    end
    val
end
(F::FourierExtension2)(x,y) = evaluate(F,x,y)

function grid_evaluate(F::FourierExtension2{T}, Ω, L::Int) where T
    # Evaluates a Fourier extension Ω ⊂ [0,1]x[0,1]
    # with coefficients c at the grid points Ω ∩ ((0:L)/L)×((0:L)/L)
    # vals_l = sum_{k,j=-n:n} c(k,j) exp(2pi*i*(k*x_l + j*y_l))

    n = div(size(F.coeffs,1)-1,2)
    if L >= 2n+1
        paddedmat = [F.coeffs[n+1:2n+1,n+1:2n+1] zeros(T,n+1,L-(2n+1)) F.coeffs[n+1:2n+1,1:n];
                                                 zeros(T,L-(2n+1),L);
                     F.coeffs[1:n,n+1:2n+1]      zeros(T,n,L-(2n+1))   F.coeffs[1:n,1:n]]
    else
        paddedmat = zeros(L,L)
        for k = 1:2n+1
            for j = 1:2n+1
                paddedmat[1+mod(k-1-n,L),1+mod(j-1-n,L)] = paddedmat[1+mod(k-1-n,L),1+mod(j-1-n,L)] + F.coeffs[k,j]
            end
        end
    end
    ifft!(paddedmat)
    indsx, indsy = grid_filter(0:1/L:1L, 0:1/L:1, Ω)
    vals = [paddedmat[indsx[k],indsy[k]] for k = 1:length(indsx)] * L^2
    indsx, indsy, vals
end


function contourf(F::FourierExtension2{T}, Ω, L) where T
    indsx,indsy,vals = grid_eval(F, Ω, L)
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
