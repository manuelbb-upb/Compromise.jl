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

promote_modulo_nothing(T1, ::Type{Nothing})=T1
promote_modulo_nothing(T1, T2)=Base.promote_type(T1, eltype(T2))
macro ignoraise(ex, loglevelex=nothing)
	has_lhs = false
	if Meta.isexpr(ex, :(=), 2)
		lhs, rhs = esc.(ex.args)
		has_lhs = true
	else
		rhs = esc(ex)
	end
	
	return quote
		ret_val = $(rhs)
		if isa(ret_val, AbstractStoppingCriterion)
			wrapped = if !isa(ret_val, WrappedStoppingCriterion)
				WrappedStoppingCriterion(ret_val, $(QuoteNode(__source__)), false)
			else
				ret_val
			end
			$(if loglevelex isa Symbol
				quote
					if !wrapped.has_logged
						stop_msg = stop_message(wrapped.crit)
						if !isnothing(stop_msg)
							@logmsg $(esc(loglevelex)) stop_msg
						end
						wrapped.has_logged = true
					end
				end
			end)
			return wrapped
		end
		$(if has_lhs
			:($(lhs) = ret_val)
		else
			:(ret_val = nothing)
		end)
	end
end

macro ignorebreak(ex)
	has_lhs = false
	if Meta.isexpr(ex, :(=), 2)
		lhs, rhs = esc.(ex.args)
		has_lhs = true
	else
		rhs = esc(ex)
	end
	
	return quote
		ret_val = $(rhs)
		do_break = isa(ret_val, AbstractStoppingCriterion)
		$(if has_lhs
			:($(lhs) = ret_val)
		else
			:(ret_val = nothing)
		end)
		do_break && break		
	end
end

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