```@meta
EditURL = "<unknown>/docs/literate_src/README.jl"
```

# Compromise.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://manuelbb-upb.github.io/Compromise.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://manuelbb-upb.github.io/Compromise.jl/dev/)
[![Build Status](https://github.com/manuelbb-upb/Compromise.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/manuelbb-upb/Compromise.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/manuelbb-upb/Compromise.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/manuelbb-upb/Compromise.jl)

## About “CoMPrOMISE”

**Co**nstrained **M**ultiobjective **Pr**oblem **O**ptimizer with **M**odel **I**nformation to **S**ave **E**valuations

This package provides a flexible first-order solver for constrained and unconstrained
nonlinear multi-objective problems.
It uses a trust region approach and either exact derivatives or local surrogate models for
a derivative-free descent.
Box constraints are respected during model construction and treated as unrelaxable.
Box constraints and linear constraints are supported and passed down to an inner LP solver.
Nonlinear constraint functions can be modelled and are dealt with by incorporating a filter.
They are *relaxable*, i.e., all other functions must be computable even when the
constraints are violated.

## Example

### Constrained Two-Parabolas Problem

First we load the optimizer package, “Compromise.jl”:

````@example README
using Compromise
````

The package exports a simple problem structure, `MutableMOP`.
As the name suggests, this is a mutable structure to define a
multi-objective optimization problem.
We can use it to set up a problem step by step.
The only information required for initialization is the
number of variables:

````@example README
mop = MutableMOP(;num_vars = 2)
````

For a `MutableMOP`, all functions are vector-to-vector.
We can define the objectives (`objectives`),
nonlinear inequality constraints (`nl_ineq_constraints`)
and nonlinear equality constraints (`nl_eq_constraints`).
By default, these fields default to `nothing`.
Alternatively, they need objects of type `Compromise.NonlinearFunction`.
We have helpers to support normal julia functions.
For example, consider this vector-to-vector function:

````@example README
function objective_function(x)
    return [
        (x[1] - 2)^2 + (x[2] - 1)^2;
        (x[1] - 2)^2 + (x[2] + 1)^2
    ]
end
````

We can easily derive the gradients, so let's also define them
manually, to use derivative-based models:

````@example README
function objective_grads(x)
    # return the transposed Jacobian, i.e., gradients as columns
    df11 = df21 = 2 * (x[1] - 2)
    df12 = 2 * (x[2] - 1)
    df22 = 2 * (x[2] + 1)
    return [ df11 df21; df12 df22 ]
end
````

To add these functions to `mop`, we call the helper method
`add_objectives` and also specify the model class to be used.
There are shorthand symbols, for example `:exact` or `taylor1`,
for objectives with known gradients.
Alternatively, give some `Compromise.AbstractSurrogateModelConfig`.
We also have to tell the optimizer about the function signature.
`func_iip=true` would imply an in-place objective with sigunature
`objective_function!(fx, x)`.
`dim_out` is a mandatory argument.

````@example README
add_objectives!(
    mop, objective_function, objective_grads, :taylor1;
    dim_out=2, func_iip=false, grads_iip=false
)
````

!!! note
    For the above objective function, it would be sensible to
    additionally have a function `objective_values_and_grads`,
    that returns the objectives and gradients at the same time.
    That is possible, `MutableMOP` has an interface for such optimizations.

We support non-convex, nonlinear constraints (as long as they are relaxable).
For example, we can constrain the problem to ℝ² without unit ball.
For demonstration purposes, use an in-place function:

````@example README
nl_ineq_function!(y, x) = y[1] = 1 - sum(x.^2)
````

Of course, that is a fairly simple constraint function.
If it was more complicated, we could be tempted to use automatic differentiation
for derivative calculations.
Instead, you can also use derivative-free models, such as
radial basis function (RBF) models.

### Excursion: RBF Kernels
By default, a cubic kernel is used, if we use the `:rbf`
option with `add_nl_ineq_constraints!`.
To use the Gaussian kernel `φ_ε(r) = \\exp(-(εr)^2)``
with fixed shape paramater `10`, do
```julia
rbf_config = RBFConfig(; kernel=GaussianKernel(10))
```
There is other cool features. If you want to vary the shape parameter
with the trust-region radius, you can give a function ``Δ ↦ ε(Δ)``
to `GaussianKernel` (as well as `InverseMultiQuadricKernel`).

````@example README
gk = GaussianKernel(del -> min(1/del, 1_000))
Compromise.RBFModels.shape_paramater(gk, 1) # equals 1.0
Compromise.RBFModels.shape_paramater(gk, 0.0005) ## equals 1000
````

For now, we stick with the fixed shape parameter and finalize
our problem:

````@example README
add_nl_ineq_constraints!(mop, nl_ineq_function!, :rbf;
    func_iip=true, dim_out=1
)
````

The `MutableMOP` is turned into a `TypedMOP` during initialization.
We can thus simply pass it to `optimize`:

````@example README
final_vals, ret = optimize(mop, [-2.0, 0.5])
````

`ret` is the return code. If no budget or time based stopping criterion
was used, then `CRITICAL` is a nice return value, indicating a first-order
critical point.

`final_vals` is a `ValueArrays` object holding values at the final iteration
site.
`final_vals.x` is the parameter vector, `final_vals.fx` has the
objective values.
The nonlinear equality constraints are `final_vals.hx`, the inequality
constraints are `final_vals.gx`...

````@example README
final_vals.x, final_vals.fx
````

### Automatic Diffentiation
There is an optional `ForwardDiff` extension.
To use a derivative-based model without specifying the gradients by-hand,
first load `ForwardDiff`.

````@example README
using ForwardDiff
````

Now, `ForwardDiffBackend` should be available:

````@example README
diff_backend = ForwardDiffBackend()
````

Setup the problem:

````@example README
mop = MutableMOP(2)
add_objectives!(mop, objective_function, :exact;
    func_iip=false, dim_out=2, backend=diff_backend
)

optimize(mop, -5 .+ 10 .* rand(2))
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

