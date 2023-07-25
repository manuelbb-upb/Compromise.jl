using Pkg

current_env = first(Base.load_path())
begin 
Pkg.activate(@__DIR__)

using Compromise
using Documenter

using Literate
src_path = joinpath(@__DIR__, "..", "src")
out_path = joinpath(@__DIR__, "src")

Literate.markdown(joinpath(src_path, "mop.jl"), out_path; execute=false, flavor=Literate.CommonMarkFlavor())
Literate.markdown(joinpath(src_path, "models.jl"), out_path; execute=false, flavor=Literate.CommonMarkFlavor())

DocMeta.setdocmeta!(Compromise, :DocTestSetup, :(using Compromise); recursive=true)

makedocs(;
    modules=[Compromise],
    authors="Manuel Berkemeier <manuelbb@mail.uni-paderborn.de> and contributors",
    repo="https://github.com/manuelbb-upb/Compromise.jl/blob/{commit}{path}#{line}",
    sitename="Compromise.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://manuelbb-upb.github.io/Compromise.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "(Dev) Notes" => "dev_notes.md",
        "Interfaces" => ["mop.md", "models.md"],
    ],
)

deploydocs(;
    repo="github.com/manuelbb-upb/Compromise.jl",
    devbranch="main",
)
Pkg.activate(current_env)

end