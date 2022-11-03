
function AZ_algorithm(A::LinearMap, Z::LinearMap, b; rank_guess::Int=20, step_1_solver=undef)
    (step_1_solver == undef) && (step_1_solver = low_rank_solver((I - A*Z')*A; rank_guess))
    x1 = step_1_solver((I - A*Z')*b)
    x2 = Z'*(b-A*x1)
    x1 + x2
end

function low_rank_solver(A::LinearMap; rank_guess::Int=20)
    N = size(A,2)
    W = rand(complex(eltype(A)),N,min(rank_guess,N)) .- 0.5
    packed_qr = qr!(Matrix(A*W))
    b -> W * (packed_qr \ b)
end

#Solves a low rank system using adaptively growing random sketches and a pivoted QR solve
# function low_rank_solve(A::LinearMap, b, rank_guess::Int=20, tol=1e-14)
#     M, N = size(A)
#     R = min(rank_guess,N)
#     W = randn(eltype(b),N,R)
#     AW = Matrix(A*W)
#     svals = svdvals!(AW)
#     new_rank_guess = max(findlast(t->(t>tol*svals[1]),svals),1)
#     while (R < new_rank_guess) && (new_rank_guess < N)
#         Wadd = randn(eltype(b),N,R)
#         W = [W Wadd]
#         AW = [AW Matrix(A*Wadd)]
#         R *= 2
#         svals = svdvals!(AW)
#         new_rank_guess = max(findlast(t->(t>tol*svals[1]),svals),1)
#     end
#     extra_needed = new_rank_guess - R
#     if extra_needed > 0
#         Wadd = randn(eltype(b),N,extra_needed)
#         W = [W Wadd]
#         AW = [AW Matrix(A*Wadd)]
#     else
#         W = W[1:N,1:new_rank_guess]
#         AW = AW[1:M,1:new_rank_guess]
#     end
#     return W * LAPACK.gelsy!(Matrix(A*W), b, tol)[1]
# end
