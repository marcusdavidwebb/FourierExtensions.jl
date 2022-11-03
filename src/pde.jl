
differentiation_matrix(n::Int, order::Int) =
    Diagonal([(π*1im*k/2)^order for k in -n:n])

function differentiation_matrix(n::Tuple{Int,Int}, order::Tuple{Int,Int})
    nx, ny = n
    d = [(2π*1im*k)^order[1]*(2π*1im*j)^order[2] for k in -nx:nx, j in -ny:ny]
    Diagonal(d[:])
end

function derivative(f::FourierExtension, order::Int = 1)
    n = (length(f.coeffs)-1) >> 1
    D = differentiation_matrix(n, order)
    FourierExtension(f.γ, D*f.coeffs)
end

function derivative(f::FourierExtension2, order::Tuple{Int,Int})
    n = (size(f.coeffs) .-1) .>> 1
    D = differentiation_matrix(n, order)
    FourierExtension2(f.Ω, reshape(D*f.coeffs[:], 2 .*n .+ 1))
end
