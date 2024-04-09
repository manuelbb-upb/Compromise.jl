include("make_literate.jl")

using Pkg

current_env = first(Base.load_path())
try 
Pkg.activate(@__DIR__)

using Compromise
using Documenter

DocMeta.setdocmeta!(
    Compromise, :DocTestSetup, :(using Compromise); recursive=true
)

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
    pagesonly=true,
    pages=[
        "Home" => "index.md",
        "Tutorials" => [
            "README" => "README.md", 
            #"stopping.md",
            #"RBF Data Sharing" => "rbf_database_callback.md"
        ],
        #"(Dev) Notes" => "dev_notes.md",
        #"Interfaces" => ["CompromiseEvaluators.md", "mop.md", "models.md"],
    ],
    warnonly = true, # TODO be more strict about this
)

deploydocs(;
    repo="github.com/manuelbb-upb/Compromise.jl",
    devbranch="main",
)
finally
Pkg.activate(current_env)
end