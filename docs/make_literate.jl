using Pkg

current_env = first(Base.load_path())
try
Pkg.activate(@__DIR__)

using Literate

out_path = joinpath(@__DIR__, "src")

# Process Literate source files
lit_src_path = joinpath(@__DIR__, "literate_src")
## README
## for docs:
Literate.markdown(joinpath(lit_src_path, "README.jl"), out_path)
## for github:
Literate.markdown(joinpath(lit_src_path, "README.jl"), joinpath(@__DIR__, "..");
    execute=false, flavor=Literate.CommonMarkFlavor()
)
#=
Literate.markdown(joinpath(lit_src_path, "rbf_database_callback.jl"), out_path; 
    flavor=Literate.DocumenterFlavor())
=#
# Process original source files containing documentation
src_path = joinpath(@__DIR__, "..", "src")
out_path = joinpath(@__DIR__, "src")

Literate.markdown(joinpath(src_path, "mop.jl"), out_path; 
    execute=false, flavor=Literate.CommonMarkFlavor())
Literate.markdown(joinpath(src_path, "surrogate.jl"), out_path; 
    execute=false, flavor=Literate.CommonMarkFlavor())
Literate.markdown(joinpath(src_path, "CompromiseEvaluators.jl"), out_path; 
    execute=false, flavor=Literate.CommonMarkFlavor())

finally
Pkg.activate(current_env)
end