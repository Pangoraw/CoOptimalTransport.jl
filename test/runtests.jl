using Test
using CoOptimalTransport

haskey(ENV, "TEST_GRAD") && include("./autodiff.jl")
include("./plancache.jl")

@testset "CoPlan size" begin
    n, d = rand(1:10), rand(1:10)
    n′, d′ = rand(1:10), rand(1:10)

    X = rand(d, n)
    X′ = rand(d′, n′)

    πᵛ, πˢ = coot(X, X′; n_iters = 2)
    @test size(πᵛ) == (d, d′)
    @test size(πˢ) == (n, n′)
end

@testset "Solve problem" begin
    x = [
        1.0 1.0 1.0
        0.0 0.0 0.0
    ]
    y = [
        0.0 0.0
        0.0 0.0
        1.0 1.0
    ]

    πᵛ, πˢ = coot(x, y; otplan_kwargs = (; ϵ = 0.01), n_iters = 50)

    @test πˢ * 3.0 ≈ fill(0.5, (3, 2))
    @test πᵛ * 2.0 ≈ [0.0 0.0 1.0; 0.5 0.5 0.0]
end
