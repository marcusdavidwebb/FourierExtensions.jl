
function AZ_algorithm(A::LinearMap, Z::LinearMap, b; rank_guess::Int=20, tol = 1e-12, maxiter::Int=100)
    IminusAZstar = I - A*Z'
    AminusAZstarA = IminusAZstar*A
    x1 = low_rank_solve(AminusAZstarA, IminusAZstar*b, rank_guess, tol)
    x2 = Z'*(b-A*x1)
    return x1 + x2
end

function low_rank_solve(A::LinearMap, b, rank_guess::Int=20, tol=1e-14)
    N = size(A,2)
    W = randn(eltype(b),N,min(rank_guess,N))
    return W * (Matrix(A*W) \ b)
end

#Solves a low rank system using adaptively growing random sketches and a pivoted QR solve
# function low_rank_solve(A::LinearMap, b, rank_guess::Int=20, tol=1e-14)
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
