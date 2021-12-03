unif_like(X, size) = fill!(similar(X, size), 1/prod(size))
unif_like(X, n::Int) = unif_like(X, (n,))

unif(n::Int) = ones(n) ./ n

squeeze(x, dims) = reshape(x, Tuple(size(x, i) for i in 1:ndims(x) if i ∉ dims))
squeeze(x) = squeeze(x, findall(==(1), size(x)))

unsqueezed_dims(s, dims) = ntuple(i -> i ∈ dims ? 1 : s[i-count(<(i), dims)], length(dims) + length(s))

"Adds extra dimensions with size 1"
unsqueeze(x::AbstractArray; dims) = reshape(x, unsqueezed_dims(size(x), dims))
