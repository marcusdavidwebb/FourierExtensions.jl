
function AZ_algorithm(A::LinearMap, Z::LinearMap, b::Vector; rank_guess::Int=20, tol=1e-12, step_1_solver=undef)
    (step_1_solver == undef) && (step_1_solver = low_rank_solver((I - A*Z')*A; rank_guess, tol))
    x1 = step_1_solver((I - A*Z')*b)
    x2 = Z'*(b-A*x1)
    x1 + x2
end

# Low rank solver which uses rank_guess
function low_rank_solver(A::LinearMap; rank_guess::Int=20, tol=1e-12)
    X = rand([-1.0+0im, 1.0+0im], size(A,2), min(rank_guess,size(A,2)))
    qrAX = qr!(Matrix(A*X))
    b -> X * (qrAX \ b)
 end

# Adaptive low rank solver. Slows things down considerably.
# function low_rank_solver(A::LinearMap; rank_guess::Int=20, tol=1e-12)
#     N = size(A,2)
#     X = rand([-1.0+0im, 1.0+0im], N, min(rank_guess,N))
#     AX = Matrix(A*X)
#     qrAX = qr(AX, ColumnNorm())
#     rank_est = rank(qrAX.R)
#     while rank_est + 10 > size(X,2)
#         Xtend = rand([-1.0+0im, 1.0+0im], N, min(size(X,2), N-size(X,2))) 
#         X = [X Xtend]
#         AX = [AX Matrix(A*Xtend)]
#         qrAX = qr(AX, ColumnNorm())
#         rank_est = rank(qrAX.R)
#     end
#     X = X[:,1:rank_est+10]
#     qrAX = qr!(AX[:,1:rank_est+10])
#     b -> X * (qrAX \ b)
# end

# Adaptive low rank solver. Slows things down considerably.
# Follows Algorithm 1 of Meier and Nakatsukasa 2021: https://arxiv.org/pdf/2105.07388.pdf
# function low_rank_solver(A::LinearMap; rank_guess::Int=20, tol=1e-12)
#     (M,N) = size(A)
#     X = rand([-1.0+0im, 1.0+0im], N, min(rank_guess,N))
#     AX = Matrix(A*X)
#     rank_est = rank_guess
#     while rank_guess ≤ N
#         ΘAX = fft!(rand([-1.0+0im, 1.0+0im], M).*AX)[rand(1:M,2rank_guess),:]
#         rank_est = findfirst(x -> x ≤ 1000*tol, svdvals!(ΘAX)[1:rank_guess])
#         (rank_est !== nothing) && break
#         extendX = rand([-1.0+0im, 1.0+0im], N, min(rank_guess,N-size(X,2)))
#         X = [X extendX]
#         AX = [AX Matrix(A*extendX)]
#         rank_guess *= 2
#     end
#     isequal(rank_est, nothing) && (rank_est = N)
#     X = X[:,1:rank_est]
#     AX = AX[:,1:rank_est]
#     qrAX = qr!(AX)
#     b -> X * (qrAX \ b)
# end