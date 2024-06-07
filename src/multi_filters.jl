using Dictionaries

# The structures we call “filter” and “population” (solution set)
# are pretty similar.
# In both cases, we want to check vectors against each other with respect to some partial
# ordering.
# A vector that is less or equal than some other vector is said to *dominate* the other 
# vector.
# We want to have the set elements adhere to the `AbstractSetElement`.
# The sets themselves subtype `AbstractNondominatedSet`.

# ## General Interface
# First, here is a very general interface, offering some nice utilities.
# The actual algorithm types are described in the next section.
abstract type AbstractSetElement end

## (mandatory)
"""
    dominates(elem_test::AbstractSetElement, elem_compare::AbstractSetElement)

Return `true`, if `elem_test` dominates (is less or equal than) `elem_compare`.
An element should not dominate itself."""
dominates(elem_test::AbstractSetElement, elem_compare::AbstractSetElement)=false

abstract type AbstractNondominatedSet end
## (optional)
_show_name(ndset::AbstractNondominatedSet)=_show_defaut_name(ndset)
## (helper -- don't overwrite)
_show_defaut_name(::SetType) where SetType <: AbstractNondominatedSet = Base.typename(SetType).name

## (derived / optional)
function Base.show(io::IO, ndset::AbstractNondominatedSet)
    print(io, "$(_show_name(ndset))(count=$(length(all_elements(ndset))))")
end

## (mandatory)
"Return an iterable object of all elements currently stored in `ndset`."
all_elements(ndset::AbstractNondominatedSet) = Any[]

## (mandatory)
"Return `true` if `elem` is marked as “stale” in `ndset`."
is_stale(ndset::AbstractNondominatedSet, elem::Any)=false
## (mandatory)
"Mark `elem` as “stale” in `ndset`."
mark_stale!(ndset::AbstractNondominatedSet, elem::Any)=nothing
## (mandatory)
"Remove mark from `elem` in `ndset`."
unmark_stale!(ndset::AbstractNondominatedSet, ::Any)=nothing

## (optional)
function nondominated_elements(ndset::AbstractNondominatedSet)
    return filter(!Base.Fix1(is_stale, ndset), all_elements(ndset))
end

## (optional)
"""
    is_dominated(elem::AbstractSetElement, ndset::AbstractNondominatedSet)

Return `true` if `elem` is dominated by any element in `ndset`."""
function is_dominated(elem::AbstractSetElement, ndset::AbstractNondominatedSet)
    is_nondominated_quicktest(elem, ndset) && return false
    for elem_nd in nondominated_elements(ndset)
        if dominates(elem_nd, elem)
            return true
        end
    end
    return false
end
## (optional)
"""
    is_nondominated_quicktest(
        elem::AbstractSetElement, ndset::AbstractNondominatedSet)

Before iterating all objects of `ndset`, perform a pre-check of `elem` and 
return `true` if `elem` is **not** dominated."""
is_nondominated_quicktest(elem::AbstractSetElement, ndset::AbstractNondominatedSet)=true

## (mandatory)
"Add `elem` to `ndset` without checking for dominance or removing any other elements."
function unconditionally_add_to_set!(
    ndset::AbstractNondominatedSet, elem::AbstractSetElement;
    kwargs...
)
    return false
end
## (optional)
"Modify `elem` before adding it to `ndset`."
function prepare_for_set!(elem::AbstractSetElement, ndset::AbstractNondominatedSet)
    return nothing
end

## (optional)
"""
    add_to_set!(ndset, elem; check_elem=true, log_level=Info, indent=0, kwargs...)

Add `elem` to `ndset` and remove dominated elements.
If `check_elem==true`, we ensure that it is not dominated by `ndset` 
before taking action."""
function add_to_set!(
    ndset::AbstractNondominatedSet, elem::AbstractSetElement;
    check_elem::Bool=true,
    log_level::LogLevel=Info,
    indent::Int=0,
    elem_identifier=nothing,
    kwargs...
)
    if check_elem
        if is_dominated(elem, ndset)
            @logmsg log_level "$(indent_str(indent)) 💀 Rejecting new element with identifier $(elem_identifier)."
            return false
        end
    end
    prepare_for_set!(elem, ndset)
    for elem_nd in nondominated_elements(ndset)
        if dominates(elem, elem_nd)
            mark_stale!(ndset, elem_nd)
            sid = get_identifier(elem_nd)
            !isnothing(sid) && @logmsg log_level "$(indent_str(indent)) 💀 Marking `$(sid)` as stale."
        end
    end
    return unconditionally_add_to_set!(ndset, elem; kwargs...)
end

# ## Algorithm Set Types
"A supertype for elements that have a vector composed of a scalar value and a vector."
abstract type AbstractAugmentedSetElement <: AbstractSetElement end
"A supertype for elements of type `AbstractAugmentedSetElement`."
abstract type AbstractAugmentedNondominatedSet <: AbstractNondominatedSet end

## (mandatory)
"Return the scalar value for `elem`."
scalar_val(elem::AbstractAugmentedSetElement) = 0
## (mandatory)
"Return the vector of values for `elem`."
vector_vals(elem::AbstractAugmentedSetElement) = []
## (optional)
"Return a human-readable identifier for `elem`."
get_identifier(elem::AbstractAugmentedSetElement) = nothing
## (optional)
"Set the identifier for `elem` to `i`."
set_identifier!(elem::AbstractAugmentedSetElement, i::Int) = nothing

## (derived)
function dominates(
    elem_test::AbstractAugmentedSetElement, elem_compare::AbstractAugmentedSetElement
)
    return augmented_dominates(
        scalar_val(elem_test), vector_vals(elem_test), 
        scalar_val(elem_compare), vector_vals(elem_compare)
    )
end

## (helper)
function augmented_dominates(scalar_test, vec_test, scalar_compare, vec_compare)
    if isempty(vec_test) && isempty(vec_compare)
        return scalar_test < scalar_compare
    end
    return (
        scalar_test <= scalar_compare && 
        all( vec_test .<= vec_compare ) && 
        (
            scalar_test < scalar_compare ||
            any( vec_test .<  vec_compare ) 
        )
    )
end

## (kind of mandatory (??))
"Delete stale elements from `ndset`."
function remove_stale_elements!(ndset::AbstractAugmentedNondominatedSet)
    return nothing
end

# ### Implementations

abstract type AbstractFilterMeta end
struct NoFilterMeta <: AbstractFilterMeta end
float_type(::AbstractFilterMeta)=DEFAULT_FLOAT_TYPE
dim_vars(::AbstractFilterMeta)=0
dim_objectives(::AbstractFilterMeta)=0
dim_nl_eq_constraints(::AbstractFilterMeta)=0
dim_nl_ineq_constraints(::AbstractFilterMeta)=0
dim_lin_eq_constraints(::AbstractFilterMeta)=0
dim_lin_ineq_constraints(::AbstractFilterMeta)=0
dim_theta(::AbstractFilterMeta)=1

Base.@kwdef struct FilterMetaMOP{F} <: AbstractFilterMeta
    float_type :: Type{F} = DEFAULT_FLOAT_TYPE
    dim_vars :: Int = 0
    dim_objectives :: Int = 0
    dim_nl_eq_constraints :: Int = 0
    dim_nl_ineq_constraints :: Int = 0
    dim_lin_eq_constraints :: Int = 0
    dim_lin_ineq_constraints :: Int = 0
end
float_type(::FilterMetaMOP{F}) where F = F
dim_vars(meta::FilterMetaMOP)=meta.dim_vars
dim_objectives(meta::FilterMetaMOP)=meta.dim_objectives
dim_nl_eq_constraints(meta::FilterMetaMOP)=meta.dim_nl_eq_constraints
dim_nl_ineq_constraints(meta::FilterMetaMOP)=meta.dim_nl_ineq_constraints
dim_lin_eq_constraints(meta::FilterMetaMOP)=meta.dim_lin_eq_constraints
dim_lin_ineq_constraints(meta::FilterMetaMOP)=meta.dim_lin_ineq_constraints

# We use this type for all of our sets:
Base.@kwdef struct AugmentedVectorFilter{
    F<:AbstractFloat, E<:AbstractAugmentedSetElement, tmpType,
    metaType<:AbstractFilterMeta
} <: AbstractAugmentedNondominatedSet
    "Dictionary of set elements."
    elems :: Dictionary{UInt64, E} = Dictionary{UInt64, AugmentedVectorElement{Float64}}()
    "Dictionary for marking elements."
    is_stale :: Dictionary{UInt64, Bool} = Dictionary{UInt64, Bool}()
    
    "Cache for minimum scalar value for quicktest."
    min_scalar_val :: Base.RefValue{F} = Ref(Inf)
    "Cache for minimum vector values for quicktest."
    min_vector_vals :: Vector{F} = Float64[]

    "Offset used with certain element types for modification."
    gamma :: F = 1e-6

    "Counter for element keys."
    counter :: Base.RefValue{Int} = Ref(0)

    "Optional cache used by some sets."
    tmp :: tmpType = nothing

    print_name :: Union{Nothing, String} = nothing

    meta :: metaType = NoFilterMeta()
end
all_elements(ndset::AugmentedVectorFilter) = values(ndset.elems)

function _show_name(ndset::AugmentedVectorFilter)
    return _show_name(ndset, ndset.print_name)
end
_show_name(ndset::AugmentedVectorFilter, ::Nothing)=_show_defaut_name(ndset)
_show_name(ndset::AugmentedVectorFilter, print_name::String)=print_name

function is_stale(ndset::AugmentedVectorFilter{F, E}, elem::E) where {F, E} 
    return get(ndset.is_stale, objectid(elem), false)
end
function mark_stale!(ndset::AugmentedVectorFilter{F, E}, elem::E) where {F, E}
    set!(ndset.is_stale, objectid(elem), true)
    nothing
end
function unmark_stale!(ndset::AugmentedVectorFilter{F, E}, elem::E) where {F, E}
    set!(ndset.is_stale, objectid(elem), false)
    nothing
end

function is_nondominated_quicktest(
    elem::AbstractAugmentedSetElement, ndset::AugmentedVectorFilter
)
    ## if `elem` is better than the utopia point, it is certainly nondominated
    vv = vector_vals(elem)
    if length(ndset.min_vector_vals) == length(vv)
        return (
            scalar_val(elem) <= ndset.min_scalar_val[] && 
            all(vector_vals(elem) .<= ndset.min_vector_vals)
        )
    end
    return true
end

function unconditionally_add_to_set!(
    ndset::AugmentedVectorFilter{F, E}, elem::E;
    elem_identifier=nothing,
    kwargs...
) where {F, E<:AbstractAugmentedSetElement} 
    
    ndset.counter[] += 1
    if isa(elem_identifier, Int) 
        ndset.counter[] = max(ndset.counter[], elem_identifier)
    end
    set_identifier!(elem, ndset.counter[])

    prepare_for_set!(elem, ndset)

    sv = scalar_val(elem)
    if sv <= ndset.min_scalar_val[]
        ndset.min_scalar_val[] = sv 
    end
    vv = vector_vals(elem)
    if length(ndset.min_vector_vals) != length(vv)
        empty!(ndset.min_vector_vals)
        append!(ndset.min_vector_vals, vv)
    else
        if all(vv .<= ndset.min_vector_vals)
            ndset.min_vector_vals .= vv
        end
    end
    k = objectid(elem)
    insert!(ndset.elems, k, elem)
    insert!(ndset.is_stale, k, false)
    return true
end

function remove_stale_elements!(ndset::AugmentedVectorFilter)
    filter!(!Base.Fix1(is_stale, ndset), ndset.elems)
    filter!(!, ndset.is_stale)
end

# Element type for the “filter” (to reduce constraint violation):
Base.@kwdef struct AugmentedVectorElement{F} <: AbstractAugmentedSetElement
    theta_fx :: Vector{F}
    key :: Base.RefValue{Int} = Ref(0)
end

scalar_val(elem::AugmentedVectorElement) = elem.theta_fx[1]
vector_vals(elem::AugmentedVectorElement) = @view(elem.theta_fx[2:end])
get_identifier(elem::AugmentedVectorElement) = elem.key[]
set_identifier!(elem::AugmentedVectorElement, i::Int) = (elem.key[] = i)

function prepare_for_set!(elem::AugmentedVectorElement, ndset::AugmentedVectorFilter)
    theta = scalar_val(elem)
    offset = ndset.gamma * theta
    elem.theta_fx .-= offset
    return nothing
end

# Element type to pre-select variable columns for the population:
Base.@kwdef struct ValueCacheElement{valsType} <: AbstractAugmentedSetElement
    vals :: valsType
    key :: Base.RefValue{Int} = Ref(0)
end
scalar_val(elem::ValueCacheElement)=cached_theta(elem.vals)
vector_vals(elem::ValueCacheElement)=cached_fx(elem.vals)
get_identifier(elem::ValueCacheElement)=elem.key[]
set_identifier!(elem::ValueCacheElement, i::Int)=elem.key[] = i

# Element type and cache container for solutions
struct SolutionStructs{
    valsType, step_valsType, step_cacheType, 
    modType, mod_valsType, iteration_scalarsType,
    statusType
} <: AbstractAugmentedSetElement
    vals :: valsType
    step_vals :: step_valsType
    step_cache :: step_cacheType
    iteration_scalars :: iteration_scalarsType
    mod :: modType
    mod_vals :: mod_valsType

    status_ref :: Base.RefValue{statusType}
    gen_id_ref :: Base.RefValue{Int}
    sol_id_ref :: Base.RefValue{Int}

    is_restored_ref :: Base.RefValue{Bool}
end
function Base.show(io::IO, sstructs::SolutionStructs)
    @unpack sol_id_ref = sstructs
    print(io, "SolutionStructs(; sol_id=$(sol_id_ref[]))")
end
scalar_val(elem::SolutionStructs) = cached_theta(elem)
vector_vals(elem::SolutionStructs) = cached_fx(elem)
get_identifier(elem::SolutionStructs) = elem.sol_id_ref[]
set_identifier!(elem::SolutionStructs, i::Int) = (elem.sol_id_ref[] = i)

is_converged(::Any)=false
is_converged(::AbstractStoppingCriterion)=true
is_converged(status_ref::Base.RefValue)=is_converged(status_ref[])
is_converged(sstructs::SolutionStructs)=is_converged(sstructs.status_ref)

@forward SolutionStructs.vals dim_vars(sols::SolutionStructs)
@forward SolutionStructs.vals dim_objectives(sols::SolutionStructs)
@forward SolutionStructs.vals dim_nl_eq_constraints(sols::SolutionStructs)
@forward SolutionStructs.vals dim_nl_ineq_constraints(sols::SolutionStructs)
@forward SolutionStructs.vals dim_lin_eq_constraints(sols::SolutionStructs)
@forward SolutionStructs.vals dim_lin_ineq_constraints(sols::SolutionStructs)
@forward SolutionStructs.vals cached_x(sols::SolutionStructs)
@forward SolutionStructs.vals cached_fx(sols::SolutionStructs)
@forward SolutionStructs.vals cached_hx(sols::SolutionStructs)
@forward SolutionStructs.vals cached_gx(sols::SolutionStructs)
@forward SolutionStructs.vals cached_ξ(sols::SolutionStructs)
@forward SolutionStructs.vals cached_Ax(sols::SolutionStructs)
@forward SolutionStructs.vals cached_Ex(sols::SolutionStructs)
@forward SolutionStructs.vals cached_Ex_min_c(sols::SolutionStructs)
@forward SolutionStructs.vals cached_theta(sols::SolutionStructs)
dim_theta(::SolutionStructs) = 1

function copy_solution_structs(sstructs)
    @unpack vals, step_vals, step_cache, iteration_scalars, mod, mod_vals, status_ref, 
        gen_id_ref, sol_id_ref, is_restored_ref = sstructs
    return SolutionStructs(
        deepcopy(vals),
        deepcopy(step_vals),
        deepcopy(step_cache),
        deepcopy(iteration_scalars),
        CE.copy_model(mod),
        deepcopy(mod_vals),
        deepcopy(status_ref),
        deepcopy(gen_id_ref),
        deepcopy(sol_id_ref),
        deepcopy(is_restored_ref)
    )
end

# ## Utility functions
# backwards-compatibility:
filter_min_theta(ndset::AugmentedVectorFilter) = ndset.min_scalar_val[]

function is_filter_acceptable(
    ndset::AugmentedVectorFilter{<:AbstractFloat, <:AugmentedVectorElement}, 
    sol::AbstractAugmentedSetElement
)
    return !is_dominated(sol, ndset)
end

function is_filter_acceptable(
    ndset::AugmentedVectorFilter{<:AbstractFloat, <:AugmentedVectorElement}, 
    vals,
)
    sol = ValueCacheElement(; vals)
    return is_filter_acceptable(ndset, sol)
end

function is_filter_acceptable(
    ndset::AugmentedVectorFilter{<:AbstractFloat, <:AugmentedVectorElement}, 
    vals, 
    vals_or_sol_augment
)
    sol = ValueCacheElement(; vals)
    return is_filter_acceptable(ndset, sol, vals_or_sol_augment)
end

function is_filter_acceptable(
    ndset::AugmentedVectorFilter{<:AbstractFloat, <:AugmentedVectorElement}, 
    sol::AbstractAugmentedSetElement,
    vals_or_sol_augment
)

    elem_augment = ndset.tmp
    elem_augment.theta_fx[1] = cached_theta(vals_or_sol_augment)
    elem_augment.theta_fx[2:end] .= cached_fx(vals_or_sol_augment)

    prepare_for_set!(elem_augment, ndset)
    if dominates(elem_augment, sol)
        return false
    end
    return is_filter_acceptable(ndset, sol)
end

function add_to_filter!(
    ndset::AugmentedVectorFilter{<:AbstractFloat, <:AugmentedVectorElement}, 
    vals_or_sol
)
    elem = make_filter_element(vals_or_sol)
    add_to_set!(ndset, elem)
end
function make_filter_element(vals_or_sol)
    theta_fx = vcat(cached_theta(vals_or_sol), cached_fx(vals_or_sol))
    return AugmentedVectorElement(; theta_fx)
end

function add_to_filter_and_mark_population!(
    ndset, population, sol;
    indent=0, log_level=Info,
    elem_identifier=nothing
)

    elem = make_filter_element(sol)
    add_to_set!(ndset, elem; check_elem=false, log_level, indent, elem_identifier)

    for other_sol in all_elements(population)
        is_stale(population, other_sol) && continue
        if dominates(elem, other_sol)   # it is important to use `elem` here, not `sol`; elem has the filter offset
            sid = get_identifier(sol)
            !isnothing(sid) && @logmsg log_level "💀 Marking $(sid) as stale."
            mark_stale!(population, other_sol)
        end
    end
    return elem
end

for (dimname, fname) in (
    (:dim_vars, :cached_x),
    (:dim_vars, :cached_ξ), 
    (:dim_objectives, :cached_fx), 
    (:dim_nl_ineq_constraints, :cached_gx), 
    (:dim_nl_eq_constraints, :cached_hx),
    (:dim_lin_ineq_constraints, :cached_Ax), 
    (:dim_lin_eq_constraints, :cached_Ex),
    (:dim_lin_ineq_constraints, :cached_Ax_min_b), 
    (:dim_lin_eq_constraints, :cached_Ex_min_c),
    (:dim_theta, :cached_theta),
)
    @eval function $(fname)(population::AugmentedVectorFilter)
        return array2mat(
            mapreduce(
                $(fname), hcat, nondominated_elements(population);
                init=Matrix{float_type(population.meta)}(undef, $(dimname)(population.meta), 0)
            )
        )
    end
end

array2mat(mat::AbstractMatrix)=mat
array2mat(mat::AbstractVector)=reshape(mat, :, 1)