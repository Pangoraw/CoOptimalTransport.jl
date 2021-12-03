using Test
using CoOptimalTransport

@testset "Ground Metric" begin
    n, d = rand(1:10), rand(1:10)
    X = rand(d, n)
    n′, d′ = rand(1:10), rand(1:10)
    Y = rand(d′, n′)
    target = []

    X[:,1] == [1., 2.]

    Δ = CoOptimalTransport.L(X, Y)

    for i ∈ axes(X, 1),
        j ∈ axes(Y, 1),
        k ∈ axes(X, 2),
        l ∈ axes(Y, 2)
        @test abs(X[i,k] - Y[j,l]) ≈ Δ[i,k,j,l]
    end
end