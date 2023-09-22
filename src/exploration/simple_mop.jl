# # Constrained Two-Parabolas Problem

# To use automatic differentiation for gradients, load `ForwardDiff`:
using ForwardDiff

# Load Optimizer package
import Compromise as C

# `C.MutableMOP` is meant to be used to set up a problem
# step by step:
MOP = C.MutableMOP(;num_vars = 2)

# All functions are vector-to-vector:
function objective_function(x)
    return [
        (x[1] - 2)^2 + (x[2]-1)^2;
        (x[1] - 2)^2 + (x[2]+1)^2
    ]
end

# Contrain the problem to ℝ² without unit ball.
# For demonstration purposes, use an in-place function.
nl_ineq_function!(y, x) = y[1] = 1 - sum(x.^2)

# Add functions to `MOP`.
# The objectives are meant to be modelled with an `RBFModel`,
# and those don't need gradients.
C.add_objectives!(MOP, objective_function, :rbf; func_iip=false, dim_out=2)

# The constraint is modelled with a Taylor Polynomial,
# so we need a backend (or compute the derivatives by hand...)
C.add_nl_ineq_constraints!(MOP, nl_ineq_function!, taylor1; 
    func_iip=true, dim_out=1, backend=C.CE.ForwardDiffBackend()
)

# The `MutableMOP` is turned into a `SimpleMOP` during initialization.
# We can thus simply pass it to `optimize`:
C.optimize(MOP, [0.0, 0.0])