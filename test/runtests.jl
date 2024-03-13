if get(ENV, "CI", false) == false
    using TestEnv, Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    TestEnv.activate()
end

using Test
using SafeTestsets
#%%
@safetestset "NonlinearParametricFunction Multi" begin
    include("multifunc.jl")
end

@safetestset "Parallel RBF Opt Shared DB" begin
    using Compromise
  
    function make_objf()
        objf_counter = Compromise.FuncCallCounter()
        return  objf_counter, function (y, x)
            y[1] = sum( (x .- 1 ).^2 )    
            y[2] = sum( (x .+ 1 ).^2 )
            Compromise.CompromiseEvaluators.inc_counter!(objf_counter)
            #show Compromise.CompromiseEvaluators.read_counter(objf_counter)
            nothing 
        end
    end
    
    ξ0 = rand(2, 10)
    opts = Compromise.ThreadedOuterAlgorithmOptions()

    mop = MutableMOP(; num_vars=2, reset_call_counters = false)
    objf_counter, o! = make_objf()
    add_objectives!(mop, o!, :rbf; dim_out=2, func_iip=true, max_func_calls=10)
    r = Compromise.optimize_with_algo(mop, opts, ξ0);

    @test objf_counter.val <= 10

    mop = MutableMOP(; num_vars=2, reset_call_counters = true )
    objf_counter, o! = make_objf()
    add_objectives!(mop, o!, :rbf; dim_out=2, func_iip=true, max_func_calls=10)
    r = Compromise.optimize_with_algo(mop, opts, ξ0);

    @test objf_counter.val <= 10 * 10

    rbf_db = Compromise.RBFModels.init_rbf_database(
        2, 2, nothing, nothing, Float64
    )
    
    cfg = RBFConfig(; database=rbf_db)

    mop = MutableMOP(; num_vars=2, reset_call_counters = false)
    objf_counter, o! = make_objf()
    add_objectives!(mop, o!, cfg; dim_out=2, func_iip=true, max_func_calls=50)
    r = Compromise.optimize_with_algo(mop, opts, ξ0);

    @test objf_counter.val <= 50

    rbf_db = Compromise.RBFModels.init_rbf_database(
        2, 2, nothing, nothing, Float64
    )
    
    cfg = RBFConfig(; database=rbf_db)

    mop = MutableMOP(; num_vars=2, reset_call_counters = true)
    objf_counter, o! = make_objf()
    add_objectives!(mop, o!, cfg; dim_out=2, func_iip=true, max_func_calls=10)
    r = Compromise.optimize_with_algo(mop, opts, ξ0);
    @test objf_counter.val <= 10 * 10

    rbf_db = Compromise.RBFModels.init_rbf_database(
        2, 2, nothing, nothing, Float64
    )
    
    cfg = RBFConfig(; database=rbf_db)

    mop = MutableMOP(; num_vars=2, reset_call_counters = true)
    objf_counter, o! = make_objf()
    add_objectives!(mop, o!, cfg; dim_out=2, func_iip=true, max_func_calls=10)
    r = Any[]
    for ξi = eachcol(ξ0)
        push!(r, optimize(mop, ξi))
    end
    @test objf_counter.val <= 10 * 10

end
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

@safetestset "Subproblems" begin
    include("suproblems.jl")
end

@safetestset "Test Problems" begin
    include("testprobs.jl")
end