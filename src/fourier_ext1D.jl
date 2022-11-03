
# Represents a function on [-1,1] by a Fourier series on [-2,2]
struct FourierExtension{T}
    coeffs :: Vector{T}
end

# Constructor
function FourierExtension(f, n::Int; oversamp=2)
    m = ceil(Int, oversamp*n)
    L = 4m
    b = complex(f.(-1:1/m:1))
    padded_data = Vector{eltype(b)}(undef,L)
    ifftplan! = plan_bfft!(padded_data)
    fftplan! = plan_fft!(padded_data)
    A = LinearMap(
        (output,x) -> fourier_ext_A!(output,x,n,m,L,ifftplan!,padded_data),
        (output,y) -> fourier_ext_Astar!(output,y,n,m,L,fftplan!,padded_data),
        2m+1,2n+1; ismutating=true)
    rank_guess = min(ceil(Int, 8*log(2n+1))+10, 2n+1)
    FourierExtension(AZ_algorithm(A, A, b; rank_guess, tol=1e-14))
end

# Adaptive Constructor
function FourierExtension(f; oversamp = 2, nmin::Int=32, nmax::Int=4096)
    n = nmin
    fval = f(0.34)
    while n <= nmax
        F = FourierExtension(f, n; oversamp)
        grid,vals = grid_eval(F,round(Int,2*oversamp*n))
        fvals = f.(grid)
        fvalsnorm = norm(fvals)
        if norm(F.coeffs)/fvalsnorm < 100 # check coefficients are not too large to be stable
            if norm(abs.(fvals - vals))/fvalsnorm < sqrt(n)*1e-14 # check residual
                if abs(F(0.34) - fval) < 1e-14 # final check at a single "random" point
                    return F
                end
            end
        end
        n *= 2
    end
    @warn "Maximum number of coefficients "*string(length(F.coeffs))*" reached in constructing FourierExtension. Try setting nmax higher."
    F
end

function fourier_ext_A!(output, x, n::Int, m::Int, L::Int, ifftplan!, padded_data)
    padded_data .= 0
    @views padded_data[1:n+1] = x[n+1:2n+1]
    @views padded_data[L-n+1:L] = x[1:n]
    ifftplan!*(padded_data)
    @views output[1:m] = padded_data[L-m+1:L]
    @views output[m+1:2m+1] = padded_data[1:m+1]
    output
end

function fourier_ext_Astar!(output, y, n::Int, m::Int, L::Int, fftplan!, padded_data)
    padded_data .= 0
    @views padded_data[1:m+1] = y[m+1:2m+1]
    @views padded_data[L-m+1:L] = y[1:m]
    fftplan!*padded_data
    @views output[1:n] .= padded_data[L-n+1:L]./L
    @views output[n+1:2n+1] .= padded_data[1:n+1]./L
    output
end

# Evaluates a Fourier extension on at point x
function (F::FourierExtension)(x)
    n = div(length(F.coeffs),2)
    sum(F.coeffs[j+n+1] * exp(π*1im*j*x/2) for j in -n:n)
end

# Evaluates a Fourier extension at grid points -1:1/m:1
function grid_eval(F::FourierExtension, m::Int)
    L = 4m
    N = length(F.coeffs)
    n = div(N,2)
    @assert  L >= N
    padded_data = [F.coeffs[n+1:N]; zeros(L-N); F.coeffs[1:n]]
    bfft!(padded_data)
    vals = padded_data[[L-m+1:L; 1:m+1]]
    -1:1/m:1, vals
end

function Plots.plot(F::FourierExtension;args...)
    plot()
    plot!(F,args...)
end

function Plots.plot!(F::FourierExtension;args...)
    m = max(1000,4*length(F.coeffs))
    grid, vals = grid_eval(F,m)
    plot(grid,real(vals), args...)
end
