using Flux

@testset "Autodiff" begin
    d, n = 5, 10
    d′, n′ = 3, 8

    W = Dense(d, d′, identity; bias = false)

    X = rand(d, n)
    X′ = rand(d′, n′)

    yˢ = 1:n .== (1:n′)'
    yᵛ = 1:d′ .== (1:d′)'

    ∇π = gradient(params(W)) do
        X̂ = W(X)
        πᵛ, πˢ = coot(X̂, X′; n_iters = 3, otplan_kwargs = (; n_iters = 3,))

        sum(@. (πᵛ * yᵛ * d′) ^ 2) +
        sum(@. (πˢ * yˢ * n) ^ 2)
    end

    # Can compute gradient with respect to π
    @test length(∇π) == 1
end
