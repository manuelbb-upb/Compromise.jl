module Compromise

# Write your package code here.
struct AlgorithmOptions
end

#=
Considerations for designing the MOP interface:
The interface should
* have abstractions that make composing an MOP from vector functions a one-liner.
* be complex enough to
  - allow for cached evaluation of sub components,
  - allow for modelling of sub components?
=#

abstract type AbstractConstraintClassification end
struct LinearEquality <: AbstractConstraintClassification end
struct LinearInequality <: AbstractConstraintClassification end
struct NonlinearEquality <: AbstractConstraintClassification end
struct NonlinearInequality <: AbstractConstraintClassification end

abstract type AbstractIndex end
abstract type AbstractScalarIndex <: AbstractIndex end
abstract type AbstractFunctionIndex <: AbstractIndex end

struct VariableIndex <: AbstractScalarIndex
    val :: Int
end

struct OutputIndex <: AbstractScalarIndex
    val :: Int
end

abstract type AbstractMetadata end
abstract type AbstractVariableMetadata <: AbstractMetadata end

struct LowerBound <: AbstractVariableMetadata end
struct UpperBound <: AbstractVariableMetadata end

abstract type MOP end

function dim_variables(::MOP)::Integer end
function dim_objectives(::MOP)::Integer end
function dim_constraints(::MOP, ::AbstractConstraintClassification)::Integer end

end
