if !get(ENV, "CI", false)
    using TestEnv, Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    TestEnv.activate()
end

using Test
using SafeTestsets
#%%
@safetestset "RBFModels" begin 
    include("rbfs.jl")
end
#%%
@safetestset "Misc" begin
    include("misc.jl")
end
@safetestset "Matrix Factorizations" begin
    include("matrix_factorizations.jl")
end