
function AZ_algorithm(A::LinearMap, Z::LinearMap, b; rank_guess::Int=20, tol = 1e-12, maxiter::Int=100, old=false)
    IminusAZstar = I - A*Z'
    AminusAZstarA = IminusAZstar*A
    if old
        x1 = old_low_rank_solve(AminusAZstarA, IminusAZstar*b, rank_guess, tol)
    else
        x1 = low_rank_solve(AminusAZstarA, IminusAZstar*b, rank_guess, tol, maxiter)
    end
    x2 = Matrix(Z'*(b-A*x1))[:]
    return x1 + x2
end

#=function low_rank_solve(A::LinearMap, b, N::Int, rank_guess::Int=20, tol = 1e-12, maxiter::Int=100)
    # SLEEPER stands for Sketch on the LEft Efficiently and PrEcondition on the Right.
    # Based on solving min||x|| s.t. Y^T(Ax-b)=0, where Y is a random matrix.
    # A blendenpik-style left preconditioner R' is used, which satisfies
    # κ(A^T Y / R) = O(1), for an LSQR iteration.
    
    extra_cols = 10
    r = min(rank_guess,N-extra_cols)
    Y = randn(eltype(b),,r)
    svals = svdvals!(Matrix{complex(eltype(b))}(A'*Y))
    new_rank_guess = max(findlast(t->(t>tol*svals[1]),svals),1)
    while (R - new_rank_guess < extra_cols) && (new_rank_guess + extra_cols < N)
        W = [W randn(eltype(b),N,R)]
        R *= 2
        svals = svdvals!(A(W))
        new_rank_guess = max(findlast(t->(t>tol*svals[1]),svals),1)
    end
    extra_needed = new_rank_guess + extra_cols - R
    if extra_needed > 0
        W = [W randn(eltype(b),N,extra_needed)]
    else
        W = W[1:N,1:new_rank_guess+extra_cols]
    end

    # Random sketch on the left of A
    M = length(b)
    Y = randn(M,rank_guess)
    AtY = Matrix{complex(eltype(b))}(A'*Y)

    # decoher the rows of A'*Y
    d = rand([-1,1],N)
    SAtY = d.*AtY/sqrt(N)
    fft!(SAtY,1)
    # take a random sample of 2r rows
    SAtY = SAtY[rand(1:N,2r),:]
    # R is such that AtY/R is well-conditioned with high prob.
    ~,R = qr!(SAtY)
     
    # Use LSQR on YtA with R' as a preconditioner
    RtinvYtA = LinearMap(x -> R' \ (AtY' * x), y -> AtY * (R \ y), N, 2r)
    x = lsqr(RtinvYtA, R'\(Y'*b), atol = tol, btol = tol, maxiter = maxiter)
    return x
end=#


function low_rank_solve(A::LinearMap, b, r::Int=20, tol = 1e-14, maxiter::Int=100)
    # SLEEPER stands for Sketch on the LEft Efficiently and PrEcondition on the Right.
    # Based on solving min||x|| s.t. Y^T(Ax-b)=0, where Y is a random matrix.
    # A blendenpik-style left preconditioner R' is used, which satisfies
    # κ(A^T Y / R) = O(1), for an LSQR iteration.

    # Random sketch on the left of A
    M, N = size(A)
    Y = randn(M,r)
    AtY = Matrix{complex(eltype(b))}(A'*Y)

    # decoher the rows of A'*Y
    d = rand([-1,1],N)
    SAtY = d.*AtY/sqrt(N)
    fft!(SAtY,1)
    # take a random sample of 2r rows
    SAtY = SAtY[rand(1:N,2r),:]
    # R is such that AtY/R is well-conditioned with high prob.
    ~,R = qr!(SAtY)

    # Use LSQR on YtA with R' as a preconditioner
    RtinvYtA = LinearMap(x -> R' \ (AtY' * x), y -> AtY * (R \ y), r, N)
    x = lsqr(RtinvYtA, R'\(Y'*b), atol = tol, btol = tol, maxiter = maxiter, conlim = Inf, verbose = false)
    return x
end

#Solves a low rank system using adaptively growing random sketches and a pivoted QR solve
# function old_low_rank_solve(A::LinearMap, b, rank_guess::Int=20, tol=1e-14)
#     M, N = size(A)
#     extra_cols = 10
#     R = min(rank_guess,N-extra_cols)
#     W = randn(eltype(b),N,R)
#     AW = Matrix(A*W)
#     svals = svdvals!(AW)
#     new_rank_guess = max(findlast(t->(t>tol*svals[1]),svals),1)
#     while (R - new_rank_guess < extra_cols) && (new_rank_guess + extra_cols < N)
#         Wadd = randn(eltype(b),N,R)
#         W = [W Wadd]
#         AW = [AW Matrix(A*Wadd)]
#         R *= 2
#         svals = svdvals!(AW)
#         new_rank_guess = max(findlast(t->(t>tol*svals[1]),svals),1)
#     end
#     extra_needed = new_rank_guess + extra_cols - R
#     if extra_needed > 0
#         Wadd = randn(eltype(b),N,extra_needed)
#         W = [W Wadd]
#         AW = [AW Matrix(A*Wadd)]
#     else
#         W = W[1:N,1:new_rank_guess+extra_cols]
#         AW = AW[1:M,1:new_rank_guess+extra_cols]
#     end
#     return W * (qr!(AW) \ b)
# end

function old_low_rank_solve(A::LinearMap, b, rank_guess::Int=20, tol=1e-14)
    ~, N = size(A)
    extra_cols = 10
    R = min(rank_guess,N-extra_cols)
    W = randn(eltype(b),N,R+extra_cols)
    return W * (qr!(Matrix(A*W)) \ b)
end