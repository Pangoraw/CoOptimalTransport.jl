module CoOptimalTransport

using LinearAlgebra
using Distances

unif(n::Int) = ones(n) ./ n
flatten(x::AbstractArray) = reshape(x, filter(!=(1), size(x)))

function sinkhorn(μ, ν, C; λ = 1.0, n_iters = 20)
    K = @. exp(-C / λ)

    u = ones(eltype(μ), size(μ))
    v = similar(ν)
    for _ in n_iters
        v = ν ./ (K' * u)
        u = μ ./ (K * v)
    end

    Diagonal(u) * K * Diagonal(v)
end

X ⊗ π = flatten(sum(X .* π; dims = (1, 2)))

"Pairwise distance on 1d variables"
function L(X, X′; metric = euclidean)
    d, n = size(X)
    d′, n′ = size(X′)

    M = pairwise(metric, reshape(X, 1, d * n), reshape(X′, 1, d′ * n′); dims = 2)
    reshape(M, d, n, d′, n′)
end
Lˢ(X, X′) = permutedims(L(X, X′), (2, 4, 1, 3)) # (n, n′, d, d′)
Lᵛ(X, X′) = permutedims(L(X, X′), (1, 3, 2, 4)) # (d, d′, n, n′)

"""
Computes the two optimal transport plan πˢ and πᵛ of size (n, n′) and (d, d′) respectively.

```julia
X = rand(5, 10)
X′ = rand(10, 23)
πᵛ, πˢ = coplan(X, X′; n_iters = 100)
```
"""
function coplan(
    X::M,
    X′::M,
    w = unif(size(X, 2)),
    w′ = unif(size(X′, 2)),
    v = unif(size(X, 1)),
    v′ = unif(size(X′, 1));
    n_iters = 20,
    tol = 0.05
) where {M<:AbstractMatrix}
    πˢ = w * w′'
    πᵛ = v * v′'

    for _ = 1:n_iters
        new_πᵛ = sinkhorn(v, v′, Lˢ(X, X′) ⊗ πˢ; λ = 1.0, n_iters = 20)
        πˢ = sinkhorn(w, w′, Lᵛ(X, X′) ⊗ πᵛ; λ = 1.0, n_iters = 20)

        err = sum(@. (new_πᵛ - πᵛ)^2)
        πᵛ = new_πᵛ

        err < tol && break
    end

    πᵛ, πˢ
end

end # module