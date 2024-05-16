include("test_preamble.jl")
#%%
@safetestset "Configs" begin
    include("algorithm_options.jl")
end

@safetestset "NonlinearParametricFunction Multi" begin
    include("multifunc.jl")
end

@safetestset "SimpleMOP and caches" begin
    include("simple_mop.jl")
end

@safetestset "Parallel RBF Opt Shared DB" begin
    include("threaded_rbf.jl")
end

@safetestset "RBFModels" begin 
    include("rbfs.jl")
end

@safetestset "Misc" begin
    include("misc.jl")
end
@safetestset "Matrix Factorizations" begin
    include("matrix_factorizations.jl")
end
#=
@safetestset "Subproblems" begin
    include("suproblems.jl")
end
=#
# TODO re-enable "Subproblems"
@safetestset "Test Problems" begin
    include("testprobs.jl")
end