using Compromise
using Documenter

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
    ],
)

deploydocs(;
    repo="github.com/manuelbb-upb/Compromise.jl",
    devbranch="main",
)
