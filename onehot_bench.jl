using Pkg
Pkg.activate(".")

using Flux, BenchmarkTools, LinearAlgebra

x = rand(Float32, 100, 100)
y = Flux.onehotbatch(1:100, 1:100)

@which x*y #*(A::AbstractArray{T,2} where T, B::Flux.OneHotMatrix) at others\Flux.jl\src\onehot.jl:37
@which Flux.mul_faster(x, y) #mul_faster(A::AbstractArray{T,2} where T, B::Flux.OneHotMatrix) at others\Flux.jl\src\onehot.jl:40
@which x*y' #*(A::AbstractArray{T,2} where T, B::AbstractArray{T,2} where T) at stdlib\v1.5\LinearAlgebra\src\matmul.jl:151
@which Flux.mul_faster(x, y') #mul_faster(A::AbstractArray{T,2} where T, B::Adjoint{Bool,var"#s126"} where var"#s126"<:Flux.OneHotMatrix) at others\Flux.jl\src\onehot.jl:53

x*y ≈ Flux.mul_faster(x, y)
@btime x*y
# 8.801 μs (13 allocations: 40.13 KiB)
@btime Flux.mul_faster(x, y)
# 2.422 μs (2 allocations: 39.14 KiB)
x*y' ≈ Flux.mul_faster(x, y')
@btime x*y'
# 838.101 μs (9 allocations: 39.50 KiB)
@btime Flux.mul_faster(x, y')
# 3.500 μs (3 allocations: 39.17 KiB)

using CUDA

cu_x = rand(CURAND.default_rng(), Float32, 100, 100)
y = Flux.onehotbatch(1:100, 1:100)

@which cu_x*y #*(A::AbstractArray{T,2} where T, B::Flux.OneHotMatrix) at others\Flux.jl\src\onehot.jl:37
@which cu_x*y' #*(A::AbstractArray{T,2} where T, B::AbstractArray{T,2} where T) at stdlib\v1.5\LinearAlgebra\src\matmul.jl:151
@which Flux.mul_faster(cu_x, y) #mul_faster(A::AbstractArray{T,2} where T, B::Flux.OneHotMatrix) at others\Flux.jl\src\onehot.jl:40
@which Flux.mul_faster(cu_x, y') #mul_faster(A::AbstractArray{T,2} where T, B::Adjoint{Bool,var"#s126"} where var"#s126"<:Flux.OneHotMatrix) at others\Flux.jl\src\onehot.jl:53

cu_x*y ≈ Flux.mul_faster(cu_x, y)
@btime CUDA.@sync cu_x*y
# 69.200 μs (76 allocations: 2.84 KiB)
@btime CUDA.@sync cpu(cu_x)*y
# 98.799 μs (30 allocations: 79.84 KiB)
@btime CUDA.@sync Flux.mul_faster(cu_x, y)
# 561.161 ms (60019 allocations: 2.44 MiB)
@btime CUDA.@sync Flux.mul_faster(cpu(cu_x), y)
# 90.300 μs (19 allocations: 78.86 KiB)
cu_x*y' ≈ Flux.mul_faster(cu_x, y')
@btime CUDA.@sync cu_x*y'
# 982.715 ms (60026 allocations: 2.48 MiB)
@btime CUDA.@sync cpu(cu_x)*y'
# 952.200 μs (26 allocations: 79.22 KiB)
@btime CUDA.@sync Flux.mul_faster(cu_x, y')
# 1.052 s (90043 allocations: 3.66 MiB)
@btime CUDA.@sync Flux.mul_faster(cpu(cu_x), y')
# 90.400 μs (20 allocations: 78.89 KiB)
