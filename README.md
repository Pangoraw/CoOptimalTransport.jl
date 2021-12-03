# CoOptimalTransport.jl

A &nbsp;
    <a href="https://julialang.org">
        <img src="https://raw.githubusercontent.com/JuliaLang/julia-logo-graphics/master/images/julia.ico" width="16em">
        Julia
    </a>
    &nbsp; implementation of CO-Optimal Transport[1] (COOT).

## Usage

### With the included entropic Sinkhorn-Knopp implementation

```julia
using CoOptimalTransport

πᵛ, πˢ = coot(
    X, X′,
    w, w′,
    v, v′;
    n_iters = 20,
    tol = 1e-4,
    metric = CoOptimalTransport.L²,
    otplan_kwargs = (; n_iters = 10, ϵ = 0.001),
    otplan = CoOptimalTransport.sinkhorn,
)
```

### With [OptimalTransport.jl](https://github.com/JuliaOptimalTransport/OptimalTransport.jl)

#### Entropic variant

```julia
using CoOptimalTransport, OptimalTransport

πᵛ, πˢ = coot(
    X, X′,
    w, w′,
    v, v′;
    n_iters = 20,
    tol = 1e-7,
    metric = CoOptimalTransport.L²,
    otplan = (μ, ν, C) -> OptimalTransport.sinkhorn(μ, ν, C, 0.001),
)
```

#### Exact variant

```julia
using CoOptimalTransport, OptimalTransport, GLPK

πᵛ, πˢ = coot(
    X, X′,
    w, w′,
    v, v′;
    n_iters = 20,
    tol = 1e-7,
    metric = CoOptimalTransport.L²,
    otplan = (μ, ν, C) -> OptimalTransport.emd(μ, ν, C, GLPK.Optimizer()),
)
```

## References

> [1]I. Redko, T. Vayer, R. Flamary, and N. Courty, “CO-Optimal Transport,” arXiv:2002.03731 [cs, stat], Nov. 2020, Accessed: Dec. 03, 2021. [Online]. Available: http://arxiv.org/abs/2002.03731
