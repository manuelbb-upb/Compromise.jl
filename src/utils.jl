ensure_float_type(::Nothing, ::Type)=nothing
function ensure_float_type(arr::AbstractArray{F}, ::Type{F}) where F
	return arr
end

function ensure_float_type(arr::AbstractArray{F}, ::Type{T}) where {F,T}
	return T.(arr)
end

function Accessors.set(
	algo_opts::Union{AlgorithmOptions{F}, SteepestDescentConfig{F}}, ::PropertyLens{:float_type}, ::Type{F}
) where F
	return algo_opts
end
function Accessors.set(
	algo_opts::Union{AlgorithmOptions{T}, SteepestDescentConfig{T}}, 
	::PropertyLens{:float_type}, 
	::Type{new_float_type}
) where {T, new_float_type}
	props = Accessors.getproperties(algo_opts)
	fnames = keys(props)
	patch = NamedTuple{fnames}(
		map(
			(k, v) -> change_float_type(v, PropertyLens(k), new_float_type), 
			fnames,
			props
		)
	)
	return Accessors.setproperties(algo_opts, patch)
end

function Accessors.set(
	algo_opts::Union{AlgorithmOptions{T}, SteepestDescentConfig{T}},
	l::PropertyLens{field}, 
	val
) where {T, field}
	return Accessors.setproperties(algo_opts, (; field => change_float_type(val, l, T)))
end


change_float_type(x, ::Type{new_float_type}) where new_float_type = x
function change_float_type(x::F, ::Type{new_float_type}) where{F<:AbstractFloat, new_float_type}
	return convert(new_float_type, x)
end
function change_float_type(x::Array{F, N}, ::Type{new_float_type}) where {F, N, new_float_type} 
	return convert(Array{new_float_type, N}, x)
end
function change_float_type(::Type{T}, ::Type{new_float_type}) where{T<:AbstractFloat, new_float_type<:AbstractFloat}
	return new_float_type
end
	
change_float_type(x, proplens, new_float_type)=change_float_type(x, new_float_type)	

function vec2str(x, max_entries=typemax(Int), digits=10)
	x_end = min(length(x), max_entries)
	_x = x[1:x_end]
	_, i = findmax(abs.(_x))
	len = length(string(trunc(_x[i]))) + digits + 1
	x_str = "[\n"
	for xi in _x
		x_str *= "\t" * lpad(string(round(xi; digits)), len, " ") * ",\n"
	end
	x_str *= "]"
	return x_str
end

function array(T::Type, size...)
  return Array{T}(undef, size...)
end

function _trial_point_accepted(iteration_status)
    return _trial_point_accepted(iteration_status.step_class)
end
function _trial_point_accepted(step_class::STEP_CLASS)
	return Int8(step_class) > 0
end

"""
	`var_bounds_valid(lb, ub)`
Return `true` if lower bounds `lb` and upper bounds `ub` are consistent.
"""
var_bounds_valid(lb, ub)=true
var_bounds_valid(lb::Nothing, ub::RVec)=!(any(isequal(-Inf), ub))
var_bounds_valid(lb::RVec, ub::Nothing)=!(any(isequal(Inf), lb))
var_bounds_valid(lb::RVec, ub::RVec)=all(lb .<= ub)

function project_into_box!(x, lb, ub)
	project_into_lower_bounds!(x, lb)
	project_into_upper_bounds!(x, ub)
end

project_into_lower_bounds!(x, ::Nothing)=nothing
project_into_lower_bounds!(x, lb)=begin
	x .= max.(x, lb)
	nothing
end
project_into_upper_bounds!(x, ::Nothing)=nothing
project_into_upper_bounds!(x, ub)=begin
	x .= min.(x, ub)
	nothing
end

function project_into_box!(x, lin_cons::LinearConstraints)
	project_into_lower_bounds!(x, lin_cons.lb)
	project_into_upper_bounds!(x, lin_cons.ub)
end

const SUPERSCRIPT_DICT = Base.ImmutableDict(
	0 => "⁰",
	1 => "¹",
	2 => "²",
	3 => "³",
	4 => "⁴",
	5 => "⁵",
	6 => "⁶",
	7 => "⁷",
	8 => "⁸",
	9 => "⁹"
)

const SUBSCRIPT_DICT = Base.ImmutableDict(
	0 => "₀",
	1 => "₁",
	2 => "₂",
	3 => "₃",
	4 => "₄",
	5 => "₅",
	6 => "₆",
	7 => "₇",
	8 => "₈",
	9 => "₉"
)

const INDENT_STRINGS = Base.ImmutableDict(
	(i => lpad("", i) for i = 0:10)...
)

function indent_str(i)
	return INDENT_STRINGS[i]
end

function supscript(num::Integer)
	return join((SUPERSCRIPT_DICT[i] for i in reverse(digits(num))), "")
end
function subscript(num::Integer)
	return join((SUBSCRIPT_DICT[i] for i in reverse(digits(num))), "")
end

function pretty_row_vec(
	x::AbstractVector;
	cutoff=80
)
	repr_str = "["
	lenx = length(x)
	for (i, xi) in enumerate(x)
		xi_str = @sprintf("%.2e", xi)
		if length(repr_str) + length(xi_str) >= cutoff
			repr_str *= "..."
			break
		end
		repr_str *= xi_str
		if i < lenx
			repr_str *= ", "
		end
	end
	repr_str *= "]"
	return repr_str
end
pretty_row_vec(x)=string(x)

universal_copy!(trgt, src)=nothing
function universal_copy!(trgt::AbstractArray{T, N}, src::AbstractArray{F, N}) where{T, F, N}
	copyto!(trgt, src)
	return nothing
end
function universal_copy!(trgt::Base.RefValue{T}, src::Base.RefValue{F}) where {T, F}
	trgt[] = src[]
	nothing
end

function custom_copy!(trgt, src)
	if objectid(trgt) == objectid(src)
		return nothing
	end
	return universal_copy!(trgt, src)
end

function universal_copy!(
	step_vals_trgt::StepValueArrays, step_vals_src::StepValueArrays
)
	for fn in fieldnames(StepValueArrays)
		trgt_fn = getfield(step_vals_trgt, fn)
		universal_copy!(trgt_fn, getfield(step_vals_src, fn))
	end
	return nothing
end

function universal_copy!(::AbstractStepCache, ::AbstractStepCache)
    error("`universal_copy!` not defined.")
end

function universal_copy!(trgt::SteepestDescentCache, src::SteepestDescentCache)
    for fn in (:fxn, :lb_tr, :ub_tr, :Axn, :Dgx_n)
        custom_copy!(
            getfield(trgt, fn),
            getfield(src, fn)
        )
    end
end

function universal_copy!(
	scaled_cons::LinearConstraints, lin_cons::LinearConstraints)
    for fn in fieldnames(LinearConstraints)
        universal_copy!(
            getfield(scaled_cons, fn), 
            getfield(lin_cons, fn)
        )
    end
    return nothing
end

function universal_copy!(
	trgt::WrappedMOPCache, src::WrappedMOPCache
)
	for fn in fieldnames(WrappedMOPCache)
		universal_copy!(
			getfield(trgt, fn), 
			getfield(src, fn)
		)
	end
end
