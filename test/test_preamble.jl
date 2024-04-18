if get(ENV, "CI", false) == false
    using TestEnv, Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    TestEnv.activate()
end

using Test
using SafeTestsets