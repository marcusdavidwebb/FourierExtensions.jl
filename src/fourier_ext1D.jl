
# Represents a function on [-1,1] by a Fourier series on [-2,2]
struct FourierExtension{T}
    coeffs :: Vector{T}
end

# Constructor
function FourierExtension(f, n::Int; oversamp=2)
    m = ceil(Int, oversamp*n)
    L = ceil(Int, 4 * m)
    N = 2n+1
    b = complex(f.((-m:m) * 4 / L) / sqrt(L))
    A = LinearMap(x -> fourier_ext_A(x,n,m,L), y -> fourier_ext_Astar(y,n,m,L), 2m+1,N)
    rank_guess = min(ceil(Int, 8*log(N))+10, N)
    coeffs = AZ_algorithm(A, A, b; rank_guess, tol=1e-14)
    return FourierExtension(coeffs)
end

# Adaptive Constructor
function FourierExtension(f; oversamp = 2, nmin::Int=32, nmax::Int=4096)
    n = nmin
    fval = f(0.34)
    while n <= nmax
        F = FourierExtension(f, n; oversamp)
        grid,vals = grid_evaluate(F,round(Int,2*oversamp*n))
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
    return F
end

function fourier_ext_A(x, n::Int, m::Int, L::Int)
    # A_{j,k} = exp(2pi*i*j*k/L)/sqrt(L) for j = -m:m, k = -n:n
    w = [x[n+1:2n+1]; zeros(complex(eltype(x)),L-2n-1); x[1:n]]
    ifft!(w)
    return [w[end-m+1:end]; w[1:m+1]] * sqrt(L)
end

function fourier_ext_Astar(y, n::Int, m::Int, L::Int)
    # A_{j,k} = exp(-2pi*i*j*k/L)/sqrt(L) for j = -n:n, k = -m:m
    w = [y[m+1:2m+1]; zeros(complex(eltype(y)),L-2m-1); y[1:m]]
    fft!(w)
    return [w[L-n+1:L]; w[1:n+1]] / sqrt(L)
end

function evaluate(F::FourierExtension, x)
    # Evaluates a Fourier extension [-1,1] ⊂ [-2,2]
    # with coefficients c at grid points x
    # vals = sum_{j=-n:n} c_j exp(π*i*j*x/γ)
    n = (length(F.coeffs)-1)>>1
    sum(F.coeffs[j+n+1] * exp(π*1im*j*x/2) for j in -n:n)
end

(F::FourierExtension)(x) = evaluate(F,x)

function grid_evaluate(F::FourierExtension, m::Int)
    # Evaluates a Fourier extension [-1,1] ⊂ [-2,2]
    # with coefficients c at grid points (-m:m)*4/L
    # vals(k) = sum_{j=-n:n} c_j exp(2pi*i*j*k/L), where L = ceil(4m).

    L = ceil(Int, 4m)
    N = length(F.coeffs)
    n = div(N-1,2)
    if  L >= N
        w = [F.coeffs[n+1:N]; zeros(L-N); F.coeffs[1:n]]
    else
        w = zeros(L)
        for j = 1:N
            w[1+mod(j-1-n,L)] = w[1+mod(j-1-n,L)] + F.coeffs[j]
        end
    end
    ifft!(w)
    grid = (-m:m)*4/L
    vals = L*w[[L-m+1:L; 1:m+1]]
    return grid, vals
end

function Plots.plot(F::FourierExtension;args...)
    plot()
    plot!(F,args...)
end

function Plots.plot!(F::FourierExtension;args...)
    m = max(1000,4*length(F.coeffs))
    grid, vals = grid_evaluate(F,m)
    plot(grid,real(vals), args...)
end
