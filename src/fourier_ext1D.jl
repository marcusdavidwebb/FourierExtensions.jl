# Represents a function on [-1,1] by a Fourier series on [-2,2]
struct FourierExtension{T}
    coeffs :: Vector{T}
end

# Constructor
function FourierExtension(f, n::Int; tol=1e-12, oversamp=2)
    m = ceil(Int, oversamp*n)
    b = complex(f.(-1:1/m:1))
    padded_data = Vector{eltype(b)}(undef,4m)
    ifftplan! = plan_bfft!(padded_data)
    fftplan! = plan_fft!(padded_data)
    A = LinearMap(
        (output,x) -> fourier_ext_A!(output,x,n,m,ifftplan!,padded_data),
        (output,y) -> fourier_ext_Astar!(output,y,n,m,fftplan!,padded_data),
        2m+1, 2n+1; ismutating=true)
    rank_guess = min(ceil(Int, 8*log(2n+1))+10, 2n+1)
    FourierExtension(AZ_algorithm(A, A, b; rank_guess, tol))
end

# Adaptive Constructor
function FourierExtension(f; tol=1e-12, oversamp=2, nmin=32, nmax=4096)
    n = nmin
    while n <= nmax
        F = FourierExtension(f, n; tol, oversamp)
        grid,vals = grid_eval(F,ceil(Int,2*oversamp*n)) # use double resolution to check error
        fvals = f.(grid)
        fvalsnorm = norm(fvals)
        if norm(F.coeffs)/fvalsnorm < 100 # check coefficients are not too large to be stable
            if norm(abs.(fvals - vals))/fvalsnorm < sqrt(n)*tol # check residual
                if (x -> abs(F(x)-f(x)))(rand()) < tol # final check at a single random point
                    return F
                end
            end
        end
        n *= 2
    end
    @warn "Maximum number of coefficients "*string(length(F.coeffs))*" reached in constructing FourierExtension. Try setting nmax higher."
    F
end

function fourier_ext_A!(output, x, n::Int, m::Int, ifftplan!, padded_data)
    padded_data .= 0
    @views padded_data[1:n+1] = x[n+1:2n+1]
    @views padded_data[end-n+1:end] = x[1:n]
    ifftplan!*(padded_data)
    @views output[1:m] = padded_data[end-m+1:end]
    @views output[m+1:2m+1] = padded_data[1:m+1]
    output
end

function fourier_ext_Astar!(output, y, n::Int, m::Int, fftplan!, padded_data)
    padded_data .= 0
    @views padded_data[1:m+1] = y[m+1:2m+1]
    @views padded_data[end-m+1:end] = y[1:m]
    fftplan!*padded_data
    @views output[1:n] .= padded_data[end-n+1:end]./length(padded_data)
    @views output[n+1:2n+1] .= padded_data[1:n+1]./length(padded_data)
    output
end

# Evaluates a Fourier extension on at point x
function (F::FourierExtension)(x)
    n = div(length(F.coeffs),2)
    real(sum(F.coeffs[j+n+1] * exp(π*1im*j*x/2) for j in -n:n))
end

# Evaluates a Fourier extension at grid points -1:1/m:1
function grid_eval(F::FourierExtension, m::Int)
    L = 4m
    n = div(length(F.coeffs),2)
    @assert  L >= 2n+1
    padded_data = [F.coeffs[n+1:2n+1]; zeros(L- 2n - 1); F.coeffs[1:n]]
    bfft!(padded_data)
    vals = real(padded_data[[L-m+1:L; 1:m+1]])
    -1:1/m:1, vals
end

function Plots.plot(F::FourierExtension;args...)
    grid, vals = grid_eval(F, max(1000,4length(F.coeffs)))
    plot(grid, vals, args...)
end
