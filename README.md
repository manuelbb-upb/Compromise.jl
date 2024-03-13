# Compromise.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://manuelbb-upb.github.io/Compromise.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://manuelbb-upb.github.io/Compromise.jl/dev/)

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
### Release Notes

I don't really keep up a consistent versioning scheme.
But the changes in this section have been significant enough to warrant some comments.

#### Version 0.0.2

##### RBF Surrogate Models
The internals of how RBF Surrogates are constructed has been redone.
As before, the construction is based off “Derivative-Free Optimization Algorithms For Computationally Expensive Functions,” (Wild, 2009).
In the old version, I did not really care for the matrix factorizations.
Finding a poised set of points for fully-linear interpolation needs repeated QR factorizations
of the point matrix.
Afterwards, additional points are found by updating the Cholesky factorization of
some symmetric matrix product involving the RBF Gram matrix.

* To save allocations, the QR factorizations now make use of structures similar to
those in `FastLapackInterface`.
  Once I manage to [make a pull request](https://github.com/DynareJulia/FastLapackInterface.jl/issues/40)
  to avoid even more allocations, we can also make `FastLapackInterface` a dependency.
* The Cholesky factorization is now updated by using the formulas from the reference,
  and the factors are used to compute the surrogate coefficients.
* In both cases, buffers are pre-allocated to support a maximum number of interpolation
  points, and we work with views mainly.
  Such temporary buffers are stored in `RBFTrainingBuffers`.
* An `RBFModel` now only needs `RBFParameters` for successful training and
  reproducible evaluation.
  Most importantly, evaluation is decoupled from the `RBFDatabase`!!
  In older versions, we would view into the database to query interpolation points.
  These are now copied instead, so that changes to the database don't invalidate a model.
* With [commit cc709fa](https://github.com/manuelbb-upb/Compromise.jl/commit/cc709fa0391d4a796b543a0733c31fb6f2e6ad46)
  we can thus share a database in multiple optimization runs.
* As an aside, there is a whole new backend for RBFs, which can be used in a standalone manner, too.

For most of the RBF related changes, [commit ab5cba8](https://github.com/manuelbb-upb/Compromise.jl/commit/ab5cba8f3d4ba39cd4bf2757a072bb655b4f0cc2)
is most relevant.

##### Other changes

* There likely was a bug in how descent directions were computed.
  In old versions, I tried to avoid the computation of an initial steplength by making it part
  of the descent direction sub-problem, but accounting for the change in criticality measure
  did not work out well.
  [Commit f1386c2](https://github.com/manuelbb-upb/Compromise.jl/commit/f1386c29e19b7d5c3bad28fc03af8024015666c5)
  makes things look a bit more elegant.
* At some point in time, I messed up affine scaling. Should work now, and there is tests for it.
* Threaded parallel execution is now supported internally (but improvised).
* Lots of tests.
* Changes to `AbstractNonlinearOperator` interface.
  A new `AbstractNonlinearOperatorWrapper <: AbstractNonlinearOperator`.
* New defaults in `AlgorithmOptions`. Stopping based mainly on minimum radius.
* A new return type (`ReturnObject`).

## Example

### Constrained Two-Parabolas Problem

This first example uses Taylor Polynomials to approximate the function locally.
For that we need a gradient function.
But we also show, how to add functions with derivative-free surrogates -- in this case
nonlinear constraints.

First we load the optimizer package, “Compromise.jl”:

````julia
using Compromise
````

The package exports a simple problem structure, `MutableMOP`.
As the name suggests, this is a mutable structure to define a
multi-objective optimization problem.
We can use it to set up a problem step by step.
The only information required for initialization is the
number of variables:

````julia
mop = MutableMOP(;num_vars = 2)
````

For a `MutableMOP`, all functions are vector-to-vector.
We can define the objectives (`objectives`),
nonlinear inequality constraints (`nl_ineq_constraints`)
and nonlinear equality constraints (`nl_eq_constraints`).
By default, these fields default to `nothing`.
Alternatively, they need objects of type `Compromise.NonlinearFunction`.
We have helpers to support normal Julia functions.
For example, consider this vector-to-vector function:

````julia
function objective_function(x)
    return [
        (x[1] - 2)^2 + (x[2] - 1)^2;
        (x[1] - 2)^2 + (x[2] + 1)^2
    ]
end
````

We can easily derive the gradients, so let's also define them
manually, to use derivative-based models:

````julia
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
We also have to tell the optimizer about the function signature.
`func_iip=true` would imply an in-place objective with signature
`objective_function!(fx, x)`.
`dim_out` is a mandatory argument.

````julia
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

````julia
nl_ineq_function!(y, x) = y[1] = 1 - sum(x.^2)
````

Of course, that is a fairly simple constraint function.
If it was more complicated, we could be tempted to use automatic differentiation
for derivative calculations.
Instead, you can also use derivative-free models, such as
radial basis function (RBF) models.

For now, we stick with the fixed shape parameter and finalize
our problem:

````julia
add_nl_ineq_constraints!(mop, nl_ineq_function!, :rbf;
    func_iip=true, dim_out=1
)
````

The `MutableMOP` is turned into a `TypedMOP` during initialization.
We can thus simply pass it to `optimize`:

````julia
ret = optimize(mop, [-2.0, 0.5])
````

`ret` is the return object.
You can query it using functions like `opt_vars` etc.
Final argument vector:

````julia
opt_vars(ret)
````

Final value vector:

````julia
opt_objectives(ret)
````

Final constraint vector:

````julia
opt_nl_ineq_constraints(ret)
````

### More RBF Options
Instead of passing `:rbf`, you can also pass an `RBFConfig`.
To use the Gaussian kernel:

````julia
cfg = RBFConfig(; kernel=GaussianKernel())
````

Or the inverse multiquadric:

````julia
cfg = RBFConfig(; kernel=InverseMultiQuadricKernel())
````

Then:

````julia
mop = MutableMOP(; num_vars=2)
add_objectives!(
    mop, objective_function, cfg; dim_out=2, func_iip=false,
)
ret = optimize(mop, [-2.0, 0.5])
````

See the docstring for more options.

#### Sharing an `RBFDatabase`
Normally, each optimization run initializes a new database.
But a database is only ever referenced.
We can thus pre-initialize a database and use it in multiple runs:

````julia
rbf_db = Compromise.RBFModels.init_rbf_database(2, 2, nothing, nothing, Float64)
cfg = RBFConfig(; database=rbf_db)
````

Set up problem:

````julia
mop = MutableMOP(; num_vars=2)
objf_counter = Ref(0)
function counted_objf(x)
    global objf_counter[] += 1
    return objective_function(x)
end
add_objectives!(
    mop, counted_objf, cfg; dim_out=2, func_iip=false,
)
````

First run:

````julia
ret = optimize(mop, [-2.0, 0.5])
objf_counter[]
````

Second run:

````julia
ret = optimize(mop, [-2.0, 0.0])
objf_counter[]
````

##### Parallelism
The RBF update algorithm has a lock to access the database in a safe way (?) when
multiple optimization runs are done concurrently.
There even is an “algorithm” for this:

````julia
X0 = [
    -2.0    -2.0    0.0
    0.5     0.0     0.0
]
opts = Compromise.ThreadedOuterAlgorithmOptions(;
    inner_opts=AlgorithmOptions(;
        stop_delta_min = 1e-8,
    )
)
rets = Compromise.optimize_with_algo(mop, opts, X0)
````

### Stopping based on Number of Function Evaluations
The restriction of evaluation budget is a property of the evaluators.
Because of this, it is not configurable with `AlgorithmOptions`.
You can pass `max_func_calls` as a keyword argument to `add_objectives!` and similar functions.
Likewise, `max_grad_calls` restricts the number of gradient calls,
`max_hess_calls` limits Hessian computations.

For historic reasons, the count is kept between runs.
To reset the count between runs (sequential or parallel), indicate it when setting up
the MOP.

````julia
mop = MutableMOP(; num_vars=2, reset_call_counters=false)   # default
add_objectives!(
    mop, objective_function, :rbf; dim_out=2, func_iip=false, max_func_calls=10
)
ret1 = optimize(mop, [-2, .5])
````

Now, there is no budget left for a second run:

````julia
ret2 = optimize(mop, [-2, -.5])
ismissing(opt_vars(ret2))
````

Here is a remedy:

````julia
mop.reset_call_counters=true
ret1 = optimize(mop, [-2, .5])
````

Now, there **is** budget left for a second run:

````julia
ret2 = optimize(mop, [-2, -.5])
!ismissing(opt_vars(ret2))
````

### Automatic Differentiation
There is an optional `ForwardDiff` extension.
To use a derivative-based model without specifying the gradients by-hand,
first load `ForwardDiff`.

````julia
using ForwardDiff
````

Now, `ForwardDiffBackend` should be available:

````julia
diff_backend = ForwardDiffBackend()
````

Set up the problem:

````julia
mop = MutableMOP(2)
add_objectives!(mop, objective_function, :exact;
    func_iip=false, dim_out=2, backend=diff_backend
)

optimize(mop, -5 .+ 10 .* rand(2))
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

