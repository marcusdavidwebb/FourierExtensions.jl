struct FourierExtFun2{T}
    # γ :: T1
    Ω   # indicator function of Ω ⊂ [0,1] × [0,1]
    coeffs :: Matrix{T}
end

function FourierExtFun2(f, Ω, n::Int; tol = 1e-8, oversamp::Real = 2.0)
    # Constants
    N = (2n+1)^2
    L = Integer(ceil(4*oversamp*n))
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

    b = complex([f(xgrid[indsx[k]],ygrid[indsy[k]]) for k = 1:M]) / L
    rank_guess = min(Integer(round(10*sqrt(N)*log10(N))+10), div(M,2))
    padded_tensor = Array{eltype(b),3}(undef, L, L, rank_guess)
    bfftplan = plan_bfft!(padded_tensor[:,:,1])
    A = LinearMap(x -> fourier_ext_2D_A!(x,n,indsx,indsy,L,bfftplan,padded_tensor), y -> fourier_ext_2D_Astar!(y,n,indsx,indsy,L,bfftplan,padded_tensor), M, N)

    coeffs = AZ_algorithm(A, A, b, rank_guess=rank_guess, tol=tol, maxiter=100) # Z = A for Fourier extensions
    return FourierExtFun2(Ω, reshape(coeffs,2n+1,2n+1))
end

# Adaptive Constructor
function FourierExtFun2(f, Ω; tol=1e-8, oversamp::Real = 2.0, nmin::Int=8, nmax::Int=32)
    n = nmin
    fval = f(0.51,0.49) # TODO: check at a point in Ω
    F = FourierExtFun2(Ω,ones(complex(typeof(fval)),1,1))
    while n <= nmax
        F = FourierExtFun2(f, Ω, n, oversamp=oversamp)
        indsx, indsy, vals = grid_evaluate(F, Ω, Integer(ceil(2*oversamp*n)))
        fvals = f.(indsx,indsy)
        fvalsnorm = norm(fvals)
        # perform 3 checks to see if converged
        if norm(F.coeffs)/fvalsnorm < 100
            if norm(abs.(fvals - vals))/fvalsnorm < 10*tol
                if abs(F(0.51,0.49) - fval) < 10*tol
                    return F
                end
            end
        end
        n *= 2
    end
    @warn "Maximum number of coefficients "*string(length(F.coeffs))*" reached in constructing FourierExtFun2. Try setting nmax higher."
    return F
end

function fourier_ext_2D_A!(c, n::Int, indsx::Vector{Int64}, indsy::Vector{Int64}, L::Int, bfftplan, padded_tensor::AbstractArray{T}) where T
    r = size(c,2)
    c = reshape(c, 2n+1, 2n+1, r)
    padded_tensor[:,:,1:r] = [c[n+1:2n+1,n+1:2n+1,:] zeros(T,n+1,L-(2n+1),r) c[n+1:2n+1,1:n,:];
                                                     zeros(T,L-(2n+1),L,r);
                              c[1:n,n+1:2n+1,:]      zeros(T,n,L-(2n+1),r)   c[1:n,1:n,:]]
    print("A norm: ")
    print(norm(padded_tensor[:,:,1:r]))
    println()
    for j = 1:r
        bfftplan*padded_tensor[:,:,j]
    end
    vals = [padded_tensor[indsx[k],indsy[k],j] for k=1:length(indsx), j=1:r]
    return vals
end

function fourier_ext_2D_Astar!(v, n::Int, indsx::Vector{Int64}, indsy::Vector{Int64}, L::Int, bfftplan, padded_tensor::AbstractArray{T}) where T
    r = size(v,2)
    padded_tensor[:,:,1:r] .= zero(T)
    for k = 1:length(indsx)
        padded_tensor[indsx[k],indsy[k],1:r] = conj.(v[k,:])
    end
    for j = 1:r
        bfftplan*padded_tensor[:,:,j] # note that fft = L * conj ∘ bfft ∘ conj
    end
    coeffs = conj.([padded_tensor[L-n+1:L,L-n+1:L,1:r] padded_tensor[L-n+1:L,1:n+1,1:r];
                    padded_tensor[1:n+1,L-n+1:L,1:r]   padded_tensor[1:n+1,1:n+1,1:r]])
    return reshape(coeffs, (2n+1)^2, r)
end

function grid_filter(xgrid::AbstractVector,ygrid::AbstractVector,Ω)
    indsx = Vector{Int64}(undef,0)
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
    return indsx, indsy
end

function evaluate(F::FourierExtFun2,x,y)
    # Evaluates a Fourier extension 
    # with coefficients c at points (x,y) ∈ [0,1]x[0,1]
    # vals_k = sum_{j=-nx:nx}sum_{l=-ny:ny} c_{j,l} exp(2pi*i*(j*x_k + l*y_k))
          
    zx = exp.(2π*im*x)
    zy = exp.(2π*im*y)
    Nx,Ny = size(F.coeffs)
    vals = F.coeffs[Nx,Ny]*ones(Bool,size(x))
    for ky = Ny-1:-1:1
        vals = vals.*zy+F.coeffs[Nx,ky]
    end
    for kx = Nx-1:-1:1
        innervals = F.coeffs[kx,Ny]*ones(Bool,size(x))
        for ky = Ny-1:-1:1
            innervals = innervals.*zy .+ F.coeffs[kx,ky]
        end
        vals = vals.*zx + innervals
    end
    vals = vals.*exp.(-π*im*(x*(Nx-1) + y*(Ny-1)))
    return vals
end

(F::FourierExtFun2)(x,y) = evaluate(F,x,y)

function grid_evaluate(F::FourierExtFun2{T}, Ω, L::Int) where T
    # Evaluates a Fourier extension Omega \subset [0,1]x[0,1]
    # with coefficients c at the grid points Omega ∩ ((0:L)/L)×((0:L)/L)
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
    return indsx, indsy, vals
end


function contourf(F::FourierExtFun2{T}, Ω, L) where T

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