module CoOptimalTransport

using LinearAlgebra

export coot

include("./utils.jl")
include("./PlanCache.jl")

"Entropy regularized Optimal Transport"
function sinkhorn(μ, ν, C; ϵ = 1.0, n_iters = 20)
    K = @. exp(-C / ϵ)

    u = ones_like(μ)
    v = similar(ν)
    for _ = 1:n_iters
        v = ν ./ (K' * u)
        u = μ ./ (K * v)
    end

    Diagonal(u) * K * Diagonal(v)
end

"""
    coot(X, X′, [w, w′, v, v′]; [n_iters = 20, tol = 1e-7, metric = CoOptimalTransport.L², otplan_kwargs = Dict{Symbol,Any}(), otplan = CoOptimalTransport.sinkhorn])

Computes the two optimal transport plan πᵛ and πˢ of size (d, d′) and (n, n′) respectively.

```julia
X = rand(5, 10);
X′ = rand(10, 23);
πᵛ, πˢ = coot(X, X′; n_iters = 100, otplan_kwargs = (; ϵ = 0.01, n_iters = 30));
```
"""
function coot(
    X,
    X′,
    w = unif_like(X, size(X, 2)),
    w′ = unif_like(X′, size(X′, 2)),
    v = unif_like(X, size(X, 1)),
    v′ = unif_like(X′, size(X′, 1));
    n_iters = 20,
    tol = 1e-7,
    metric = L²,
    otplan_kwargs = Dict{Symbol,Any}(),
    otplan = (args...) -> sinkhorn(args...; otplan_kwargs...),
)
    πˢ = w * w′'
    πᵛ = v * v′'

    LXX′ˢ = metric(X, X′, w, w′)
    LXX′ᵛ = metric(X', X′', v, v′)
    cost = Inf
    for _ = 1:n_iters
        Mᵛ = LXX′ˢ ⊗ πˢ
        new_πᵛ = otplan(v, v′, Mᵛ)
        Mˢ = LXX′ᵛ ⊗ new_πᵛ
        new_πˢ = otplan(w, w′, Mˢ)

        newcost = sum(@. (πᵛ * Mᵛ))
        Δ = sum(@. (πᵛ - new_πᵛ)^2) + sum(@. (πˢ - new_πˢ)^2)
        πᵛ = new_πᵛ
        πˢ = new_πˢ

        if Δ < 1e-16 || abs(cost - newcost) < tol
            break
        end

        cost = newcost
    end

    πᵛ, πˢ
end

end # module
