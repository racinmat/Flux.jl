using Pkg
Pkg.activate(".")

using Flux, BenchmarkTools, LinearAlgebra
x = rand(Float32, 100, 50)
y = Flux.onehotbatch(ones(100), 1:50)
x*y ≈ Flux.mul_faster(x, y)
@btime x*y
@btime Flux.mul_faster(x, y)

@which x*y
@which Flux.mul_faster(x, y)
@which x*y'
@which Flux.mul_faster(x, y')

x = rand(Float32, 100, 100)
y = Flux.onehotbatch(1:100, 1:100)
x*y ≈ Flux.mul_faster(x, y)
@btime x*y
@btime Flux.mul_faster(x, y)
x*y' ≈ Flux.mul_faster(x, y')
@btime x*y'
@btime Flux.mul_faster(x, y')

using CUDA
CUDA.devices() |> collect

cu_x = rand(CURAND.default_rng(), Float32, 100, 100)
y = Flux.onehotbatch(1:100, 1:100)

CUDA.allowscalar(false)

@which cu_x*y
@which cu_x*y'
@which Flux.mul_faster(cu_x, y)
@which Flux.mul_faster(cu_x, y')

cu_x*y' ≈ Flux.mul_faster(cu_x, y')
@btime CUDA.@sync cu_x*y
@btime CUDA.@sync Flux.mul_faster(cu_x, y)
cu_x*y' ≈ Flux.mul_faster(cu_x, y')
@btime CUDA.@sync cu_x*y'
@btime CUDA.@sync Flux.mul_faster(cu_x, y')
