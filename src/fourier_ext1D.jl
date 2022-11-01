
# Represents a function on [-1,1] by a Fourier series on [-γ,γ] where γ > 1
struct FourierExtension{T1,T2}
    γ :: T1
    coeffs :: Vector{T2}
end

# Constructor
function FourierExtension(f, n::Int; γ::Real=2.0, oversamp::Real = 2.0, old=false)
    m = Integer(ceil(oversamp*n))
    L = Integer(ceil(2γ * m))
    N = 2n+1
    b = complex(f.((-m:m) * 2γ / L) / sqrt(L))
    A = LinearMap(x -> fourier_ext_A(x,n,m,L), y -> fourier_ext_Astar(y,n,m,L), 2m+1,N)
    rank_guess = min(Integer(ceil(8*log(N)))+10, N)
    coeffs = AZ_algorithm(A, A, b, rank_guess=rank_guess, tol=1e-14, maxiter=100, old=old)
    return FourierExtension(γ, coeffs)
end

# Adaptive Constructor
function FourierExtension(f; γ::Real=2.0, oversamp::Real = 2.0, nmin::Int=32, nmax::Int=4096, old=false)
    n = nmin
    fval = f(0.34)
    F = FourierExtension(γ,[complex(fval)])
    while n <= nmax
        F = FourierExtension(f, n, γ = γ, oversamp=oversamp, old=old)
        grid,vals = grid_evaluate(F,Integer(round(2*oversamp*n)))
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
    # Fast matrix vector product of Fourier LS collocation matrix
    # A_{j,k} = exp(2pi*i*j*k/L)/sqrt(L) for j = -m:m, k = -n:n
    # to the column vector x (of length 2n+1).
    w = [x[n+1:2n+1,:]; zeros(complex(eltype(x)),L-2n-1,size(x,2)); x[1:n,:]]
    ifft!(w,1)
    return w[[end-m+1:end; 1:m+1],:] * sqrt(L)
end

function fourier_ext_Astar(y, n::Int, m::Int, L::Int)
    # Fast matrix vector product of adjoint of Fourier LS collocation matrix
    # A_{j,k} = exp(-2pi*i*j*k/L)/sqrt(L) for j = -n:n, k = -m:m
    # to the vector y (of length 2m+1).
    w = [y[m+1:2m+1,:]; zeros(complex(eltype(y)),L-2m-1,size(y,2)); y[1:m,:]]
    fft!(w,1)
    return w[[L-n+1:L; 1:n+1],:] / sqrt(L)
end

function evaluate(F::FourierExtension,x)
    # Evaluates a Fourier extension [-1,1] ⊂ [-γ,γ]
    # with coefficients c at grid points x
    # vals = sum_{j=-n:n} c_j exp(π*i*j*x/γ)
    z = exp.(π*1im*x/F.γ)
    N = length(F.coeffs)
    vals = F.coeffs[N]*ones(size(x))
    for k = N-1:-1:1
        vals = vals.*z .+ F.coeffs[k]
    end
    vals = vals.*exp.(-π*1im*x*(N-1)/(2*F.γ))
    return vals
end

(F::FourierExtension)(x) = evaluate(F,x)

function grid_evaluate(F::FourierExtension, m::Int)
    # Evaluates a Fourier extension [-1,1] ⊂ [-γ,γ]
    # with coefficients c at grid points (-m:m)*2γ/L
    # vals(k) = sum_{j=-n:n} c_j exp(2pi*i*j*k/L), where L = ceil(2γm).
    
    L = Integer(ceil(F.γ*2*m))
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
    grid = (-m:m)*2*F.γ/L
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