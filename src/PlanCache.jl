"""
Computes a faster L(C,C̄) ⊗ γ using the method presented in[1].
A PlanCache can be built either using L² or KL, a plan can be applied by using ⊗.

[1]G. Peyré, M. Cuturi, et J. Solomon, « Gromov-Wasserstein Averaging of Kernel and Distance Matrices »,
   in Proceedings of The 33rd International Conference on Machine Learning, june 2016, p. 2664‑2672.
"""
struct PlanCache{A<:AbstractMatrix,B<:AbstractMatrix,C<:AbstractMatrix}
    c::A
    h₁C::B
    h₂C̄::C
end

ones_like(X, dims) = fill!(similar(X, dims), one(eltype(X)))
ones_like(X, dims...) = ones_like(X, (dims...,))
ones_like(X) = ones_like(X, size(X))

_build_c(X::Matrix{Float64}, X′::Matrix{Float64}, v::Vector{Float64}, v′::Vector{Float64}, f₁::Function, f₂::Function) =
    repeat(f₁.(X) * v, inner = (1, size(X′, 1))) +
    repeat(v′', inner = (size(X, 1), 1)) * f₂.(X′)'

# repeat is faster but there is no implementation
# of Base.repeat for CuArrays yet and diffentiation does not work
# through repeat either
_build_c(X, X′, v, v′, f₁, f₂) =
    f₁.(X) * v * ones_like(v, (1, size(X′, 1))) +
    (v′ * ones_like(v′, (1, size(X, 1))))' * f₂.(X′)'

function PlanCache(X, X′, v, v′, f₁, f₂, h₁, h₂)
    c = _build_c(X, X′, v, v′, f₁, f₂)

    h₁C = h₁.(X)
    h₂C̄ = h₂.(X′)

    PlanCache(c, h₁C, h₂C̄)
end

pow2(a) = a^2
aloga(a) = a * log(a) - a

L²(X, X′, v = unif_like(X, size(X, 2)), v′ = unif_like(X′, size(X′, 2))) =
    PlanCache(X, X′, v, v′, pow2, pow2, identity, x -> 2x)
KL(X, X′, v = unif_like(X, size(X, 2)), v′ = unif_like(X′, size(X′, 2))) =
    PlanCache(X, X′, v, v′, aloga, identity, identity, log)

p::PlanCache ⊗ γ::AbstractMatrix = p.c - p.h₁C * γ * p.h₂C̄'