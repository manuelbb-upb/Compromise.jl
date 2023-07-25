# Compromise.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://manuelbb-upb.github.io/Compromise.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://manuelbb-upb.github.io/Compromise.jl/dev/)
[![Build Status](https://github.com/manuelbb-upb/Compromise.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/manuelbb-upb/Compromise.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/manuelbb-upb/Compromise.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/manuelbb-upb/Compromise.jl)

## CoMPrOMISE
**Co**nstrained **M**ultiobjective **Pr**oblem **O**ptimizer with **M**odel **I**nformation to **S**ave **E**valuations

This package provides a fExible first-order solver for constrained and unconstrained nonlinear multiobjective problems.
It uses a trust region approach and either exact derivatives or local surrogate models for a derivative-free descent.
Box constraints are respected during model construction and treated as unrelaxable.
Box constraints and linear constraints are supported and passed down to an inner LP solver.
Nonlinear constraint functions can be modelled and are dealt with by incorporating a filter. 
They are *relaxable*, i.e., all other functions must be evaluable even when the constraints are violated.

## Example

### Constrained Two-Parabolas Problem

To use automatic differentiation for gradients, load `ForwardDiff`:
```julia
using ForwardDiff
```

Load Optimizer package
```julia
import Compromise as C
```

`C.MutableMOP` is meant to be used to set up a problem
step by step:
```julia
MOP = C.MutableMOP(;num_vars = 2)
```

All functions are vector-to-vector:
```julia
function objective_function(x)
    return [
        (x[1] - 2)^2 + (x[2]-1)^2;
        (x[1] - 2)^2 + (x[2]+1)^2
    ]
end
```

Contrain the problem to ℝ² without unit ball:
```julia
function nl_ineq_function(x)
    return 1 - sum(x.^2)
end
```
Add functions to `MOP`.
The objectives are meant to be modelled with an `RBFModel`,
and those don't need gradients.
```julia
C.add_objectives!(MOP, objective_function, :rbf; func_iip=false, dim_out=2)
```

The constraint is modelled with a Taylor Polynomial,
so we need a backend (or compute the derivatives by hand...)
```julia
C.add_nl_ineq_constraints!(MOP, nl_ineq_function, :taylor1; 
    func_iip=false, dim_out=1, backend=C.CE.ForwardDiffBackend()
)
```
The `MutableMOP` is turned into a `SimpleMOP` during initialization.
We can thus simply pass it to `optimize`:
```julia
C.optimize(MOP, [-2.0, 0.5])
```
