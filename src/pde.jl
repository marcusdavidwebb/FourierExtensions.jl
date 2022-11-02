
differentiation_matrix(n::Int, order::Int, γ) =
    Diagonal([π*1im*k/γ for k in -n:n])

function derivative(f::FourierExtension)
    n = (length(f.coeffs)-1) >> 1
    D = differentiation_matrix(n, 1, f.γ)
    FourierExtension(f.γ, D*f.coeffs)
end
