# Represents a function on [-1,1] by a Fourier series on [-2,2]
struct FourierExtension
    coeffs :: Vector{ComplexF64}
end

# Constructor
function FourierExtension(f::Function, n::Int; tol = 1e-12, oversamp=2)
    m = ceil(Int, oversamp*n)
    b = complex(f.(-1:1/m:1))
    padded_data = Vector{ComplexF64}(undef,4m)
    ifftplan! = plan_bfft!(padded_data)
    fftplan! = plan_fft!(padded_data)
    A = LinearMap(
        (output,x) -> fourier_ext_A!(output, x, n, m, ifftplan!, padded_data),
        (output,y) -> fourier_ext_Astar!(output, y, n, m, fftplan!, padded_data),
        2m+1, 2n+1; ismutating=true)
    rank_guess = min(ceil(Int, 8*log(2n+1))+10, 2n+1)
    FourierExtension(AZ_algorithm(A, A/4m, b; rank_guess, tol))
end

# Adaptive Constructor
function FourierExtension(f::Function; tol=1e-12, oversamp=2, nmin=32, nmax=4096)
    n = nmin
    while n <= nmax
        global F = FourierExtension(f, n; oversamp)
        grid, vals = grid_eval(F, ceil(Int, 2*oversamp*n)) # use double resolution to check error
        fvals = f.(grid)
        fvalsnorm = norm(fvals)
        if sqrt(length(fvals))*norm(F.coeffs)/fvalsnorm < 200 # check coefficients are not too large to be stable
            if norm(fvals - vals)/fvalsnorm < sqrt(n)*tol # check residual
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

function fourier_ext_A!(output::AbstractVector, x::AbstractVector, n::Int, m::Int, ifftplan!, padded_data)
    padded_data .= 0
    @views padded_data[1:n+1] = x[n+1:2n+1]
    @views padded_data[end-n+1:end] = x[1:n]
    ifftplan!*(padded_data)
    @views output[1:m] = padded_data[end-m+1:end]
    @views output[m+1:2m+1] = padded_data[1:m+1]
    output
end

function fourier_ext_Astar!(output::AbstractVector, y::AbstractVector, n::Int, m::Int, fftplan!, padded_data)
    padded_data .= 0
    @views padded_data[1:m+1] = y[m+1:2m+1]
    @views padded_data[end-m+1:end] = y[1:m]
    fftplan!*padded_data
    @views output[1:n] .= padded_data[end-n+1:end]
    @views output[n+1:2n+1] .= padded_data[1:n+1]
    output
end

# Evaluates a Fourier extension on at point x
function (F::FourierExtension)(x)
    n = div(length(F.coeffs),2)
    real(sum(F.coeffs[j+n+1] * exp(j*x*π*0.5im) for j ∈ -n:n))
end

# Evaluates a Fourier extension at grid points -1:1/m:1. Throws error if 4m < 2n+1.
function grid_eval(F::FourierExtension, m::Int)
    n = div(length(F.coeffs),2)
    padded_data = [F.coeffs[n+1:2n+1]; zeros(ComplexF64, 4m-2n-1); F.coeffs[1:n]]
    bfft!(padded_data)
    vals = real(padded_data[[3m+1:4m; 1:m+1]])
    -1:1/m:1, vals
end

Plots.plot(F::FourierExtension; args...) = plot(grid_eval(F, 4max(100,length(F.coeffs))), args...)
