import CoOptimalTransport: L², unif, ⊗

"Baseline comparison to the PlanCache method"
function naive_l²(X, X′, γ)
    l²(a, b) = (a - b)^2
    d, n = size(X)
    d′, n′ = size(X′)

    Δ = zeros(d, d′, n, n′)
    for i = 1:d,
        j = 1:d′

        for k = 1:n,
            l = 1:n′

            Δ[i, j, k, l] = l²(X[i, k], X′[j, l]) * γ[k, l]
        end
    end

    reshape(sum(Δ; dims = (3, 4)), d, d′)
end

@testset "PlanCache" begin
    X = rand(10, 3)
    X′ = rand(5, 23)

    v = unif(size(X, 2))
    v′ = unif(size(X′, 2))

    γ = v * v′'

    LXX′ = L²(X, X′)

    time_cache = @elapsed LXX′ ⊗ γ
    time_naive = @elapsed naive_l²(X, X′, γ)

    @test time_cache < time_naive

    res_cache = LXX′ ⊗ γ
    res_naive = naive_l²(X, X′, γ)

    @test res_cache ≈ res_naive
end