# not part of the package, but script to test out things...
import DiffResults as DR
import ForwardDiff as FD
using Test
#%%
ros(x, a, b) = b*(x[2]-x[1]^2)^2 + (a - x[1])^2
∇ros(x, a, b) = [
    -4*b*(x[2]-x[1]^2)*x[1]-2*(a-x[1]);
    2*b*(x[2]-x[1]^2)
]
Hros(x, a, b) = [
    8*b*x[1]^2-4*b*(x[2]-x[1]^2)+2   -4*b*x[1];
    -4*b*x[1]                          2*b
]

function ros!(y, x)
    y[1] = ros(x, 100, 1)
    y[2] = ros(x, 50, 2)
    y[3] = ros(x, 25, 4)
    return nothing
end

x = zeros(2)
y = zeros(3)
Dy = zeros(3, 2)
Hy = zeros(6, 2)
res = DR.DiffResult(y, Dy, Hy)
#%%
import DiffResults
import DiffResults: DiffResult
import ForwardDiff: jacobian!, hessian!, HessianConfig, checktag
import ForwardDiff: AbstractConfig, JacobianConfig, Chunk
import ForwardDiff: jacobian, Dual, Tag, value

struct VectorHessianConfig{T, V, N, DI, DO}
    inner_config::JacobianConfig{T, Dual{T, V, N}, N, DI}
    outer_config::JacobianConfig{T, V, N, DO}
end

function VectorHessianConfig(
    f::F, 
    result::DiffResult,
    x::AbstractArray{V},
    chunk::Chunk = Chunk(x),
    tag = Tag(f, V)
) where {F, V}
    y = DiffResults.value(result)
    Dy = DiffResults.jacobian(result)
    nout, nvars = size(Dy)

    outer_config = JacobianConfig((f, jacobian), Dy, x, chunk, tag)
    xdual = outer_config.duals[2]
    ydual = similar(xdual, nout)
    inner_config = JacobianConfig(f, ydual, xdual, chunk, tag)
    return VectorHessianConfig(inner_config, outer_config)
end
#%%
mutable struct InnerJacobianForHess!{R,C,F}
    result::R
    cfg::C
    f::F
end

function (g::InnerJacobianForHess!)(Dydual, xdual)
    cfg = g.cfg.inner_config
    y0 = zeros(eltype(Dydual), size(Dydual,1))
    jacobian!(Dydual, g.f, y0, xdual, cfg, Val{false}())
    #g.result = DiffResults.value!(g.result, value.(DiffResults.value(inner_result)))
    g.result = DiffResults.value!(g.result, value.(y0))
    return nothing
end

function hessian!(
    result::DiffResult, 
    f::F, 
    Df::AbstractArray, 
    x::AbstractArray, 
    cfg::VectorHessianConfig{T} = VectorHessianConfig(f, result, x), ::Val{CHK}=Val{true}()
) where {F,T,CHK}
    CHK && checktag(T, f, x)
    ∇f! = InnerJacobianForHess!(result, cfg, f)
    jacobian!(DiffResults.hessian(result), ∇f!, DiffResults.jacobian(result), x, cfg.outer_config, Val{false}())
    return ∇f!.result
end

#%%
function ros(x)
    return [ros(x, 100, 1), ros(x, 50, 2), ros(x, 25, 4)]
end

function grads_ros(x)
    return hcat(
        ∇ros(x, 100, 1),
        ∇ros(x, 50, 2),
        ∇ros(x, 25, 4),
    )
end

function hess_ros(x)
    cat(
        Hros(x, 100, 1),
        Hros(x, 50, 2),
        Hros(x, 25, 4);
        dims=3
    )
end

function ros!(y, x)
    y[1] = ros(x, 100, 1)
    y[2] = ros(x, 50, 2)
    y[3] = ros(x, 25, 4)
    return nothing
end

function grads_ros!(Dy, y, x)
    FD.jacobian!(transpose(Dy), ros!, y, x)
    return nothing
end

x0 = rand(2)
y1 = Real[0.0, 0.0, 0.0]
Dy1 = zeros(2, 3)
grads_ros!(Dy1, y1, x0)

@test all( y1 .≈ ros(x0) )
@test all( Dy1 .≈ grads_ros(x0) )

# `FD.jacobian` treats colums of matrices as outputs and stacks their jacobians
# vertically.
# Suppose `grads` returns `Y`,  the transposed Jacobian of dimensions `nvars` × `nout`.  
# Calling `FD.jacobian` on `grads` will return `H` of dimensions `nout*nvars` × `nvars`.
# The first `nvars` rows constitute the Jacobian of the first column of `Y`, i.e., the 
# Jacobian of the first gradient, i.e., the transpose of the first Hessian.

# Suppose `H` is a `nvars`×`nvars`×`nout` matrix meant to store the hessians of the outputs.
# We can pass `transpose(reshape(H, nvars, nvars*nout))` to `FD.jacobian!` to get the correct
# jacobians.

function jac_ros!(Dy, x)
    global y1
    FD.jacobian!(Dy, ros!, y1, x)
    return nothing
end

H1 = zeros(2,2,3)
h1 = transpose(reshape(H1, 2, 6))
h1 = zeros(6,2)
FD.jacobian!(h1, jac_ros!, zeros(3,2), x0)

function do_all(y, Dy, H, func, x, p)
    nvars, _, nout = size(H)
    y_eps = zeros(Real, nout) #zeros(FD.Dual{eltype(y)}, nout)
    FD.jacobian!(
        transpose(reshape(H, nvars, nvars*nout)),
        (_Dy, _x) -> FD.jacobian!(
            FD.DiffResults.DiffResult(y_eps, transpose(_Dy)), (y, x)->func(y, x, p), y_eps, _x
            ), Dy, x )
    #FD.extract_value!(eltype(y), nothing, y, y_eps) 
    return y_eps
end

ros!(y, x, p) = ros!(y, x)
y2 = zeros(3)
Dy2 = zeros(2,3)
H2 = zeros(2,2,3)
yd = do_all(y2, Dy2, H2, ros!, x0, [])