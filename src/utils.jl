ensure_float_type(::Nothing, ::Type)=nothing
function ensure_float_type(arr::AbstractArray{F}, ::Type{F}) where F
	return arr
end

function ensure_float_type(arr::AbstractArray{F}, ::Type{T}) where {F,T}
	return T.(arr)
end

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

function _parse_ignoraise_expr(ex)
	has_lhs = false
	if Meta.isexpr(ex, :(=), 2)
		lhs, rhs = esc.(ex.args)
		has_lhs = true
	else
		lhs = nothing	# not really necessary
		rhs = esc(ex)
	end
	return has_lhs, lhs, rhs
end

"""
	@ignoraise a, b, c = critical_function(args...) [indent=0]

Evaluate the right-hand side.
If it returns an `AbstractStoppingCriterion`, make sure it is wrapped and return it.
Otherwise, unpack the returned values into the left-hand side.

Also valid:
```julia
@ignoraise critical_function(args...)
@ignoraise critical_function(args...) indent_var
```
The indent expression must evaluate to an `Int`.
"""
macro ignoraise(ex, indent_ex=0)
	has_lhs, lhs, rhs = _parse_ignoraise_expr(ex)
	
	return quote
		ret_val = $(rhs)
		if ret_val isa AbstractStoppingCriterion
			return wrap_stop_crit(
				ret_val, $(QuoteNode(__source__)), $(esc(indent_ex))
			)
		end
		$(if has_lhs
			:($(lhs) = wrapped)
		else
			:(ret_val = nothing)
		end)
	end
end

"""
	@ignorebreak ret_var = critical_function(args...)

Similar to `@ignoraise`, but instead of returning if `critical_function` returns 
an `AbstractStoppingCriterion`, we `break`.
This allows for post-processing before eventually returning.
`ret_var` is optional, but in constrast to `@ignoraise`, we unpack unconditionally,
so length of return values should match length of left-hand side expression.
"""
macro ignorebreak(ex, indent_ex=0)
	has_lhs, lhs, rhs = _parse_ignoraise_expr(ex)
		
	return quote
		ret_val = $(rhs)
		ret_val = wrap_stop_crit(
			ret_val, $(QuoteNode(__source__)), $(esc(indent_ex)))
		do_break = ret_val isa AbstractStoppingCriterion
		$(if has_lhs
			:($(lhs) = ret_val)
		else
			:(ret_val = nothing)
		end)
		do_break && break		
	end
end

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
	(i => lpad("", i) for i = 1:10)...
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