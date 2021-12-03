import Pkg

Pkg.activate(temp=true)
Pkg.add(["MLDatasets", "GLPK", "Colors", "ImageInTerminal", "CodecBzip2", "TranscodingStreams"])
Pkg.develop(path=abspath(joinpath(@__DIR__, "..")))

using CoOptimalTransport
using MLDatasets: MNIST
import DelimitedFiles
import Downloads
import CodecBzip2, TranscodingStreams
using Colors
import ImageInTerminal

function get_usps(usps_path=joinpath(@__DIR__, "..", "./usps"))
  if !isfile(usps_path)
      archive_file = usps_path * ".bz2"
      Downloads.download("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2", archive_file)
      data = read(archive_file)
      stream = TranscodingStreams.transcode(CodecBzip2.Bzip2Decompressor, data)
      write(usps_path, stream)
      rm(archive_file)
  end

  f = DelimitedFiles.readdlm(usps_path)
  X = parse.(Float64, split.(f[:,begin+1:end], ":") .|> last)'
  Y = Int.(f[:,begin])

  return X, Y
end

n_usps = 5_000
n_mnist = 10_000

X_usps, Y_usps = get_usps()
@assert(0 < n_usps <= size(X_usps, 2))
X_usps, Y_usps = X_usps[:,begin:n_usps], Y_usps[begin:n_usps]

X_mnist, Y_mnist = MNIST.traindata()
@assert(0 < n_mnist <= size(X_mnist, 3))
X_mnist = reshape(X_mnist, size(X_mnist, 1) * size(X_mnist, 2), size(X_mnist, 3)) .|> Float64
X_mnist, Y_mnist = X_mnist[:,begin:n_mnist], Y_mnist[begin:n_mnist]

v_usps = sum(eachcol(X_usps))
v_usps = v_usps .- minimum(v_usps)
v_usps = v_usps ./ sum(v_usps)

v_mnist = sum(eachcol(X_mnist))
v_mnist = v_mnist .- minimum(v_mnist)
v_mnist = v_mnist ./ sum(v_mnist)

n_usps = size(X_usps, 2)
n_mnist = size(X_mnist, 2)

w_mnist = fill(1/n_mnist, n_mnist)
w_usps = fill(1/n_usps, n_usps)

πᵛ, πˢ = CoOptimalTransport.coot(
    X_mnist, X_usps,
    w_mnist, w_usps,
    v_mnist, v_usps,
    otplan_kwargs = (; ϵ = .01, n_iters = 100),
)

mnist_sample = reshape(X_mnist[:,begin], 28, 28)' .|> Gray
ImageInTerminal.imshow(mnist_sample)

println()

# usps_sample = reshape(X_usps[:,begin], 16, 16)' .|> Gray
# ImageInTerminal.imshow(usps_sample)

mnist_sample_in_usps = reshape((reshape(Float64.(mnist_sample), 28 * 28)' * πᵛ) .* (28 ^ 2), 16, 16) .|> Gray
ImageInTerminal.imshow(mnist_sample_in_usps)

nothing
