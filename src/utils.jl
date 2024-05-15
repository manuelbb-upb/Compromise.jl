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

function array(::Type{T}, size::Integer) where T
  return Array{T}(undef, size)::Array{T, 1}
end
function array(::Type{T}, size::NTuple{N, <:Any}) where {T, N}
	return Array{T}(undef, size)::Array{T, N}
end
function array(T, size...)
	return array(T, size)
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

function log_stop_code(crit, log_level)
    @logmsg log_level stop_message(crit)
end


function compatibility_test_rhs(c_delta, c_mu, mu, Δ)
    return c_delta * min(Δ, c_mu + Δ^(1+mu))
end

function compatibility_test(n, c_delta, c_mu, mu, Δ)
    return LA.norm(n, Inf) <= compatibility_test_rhs(c_delta, c_mu, mu, Δ)
end

function compatibility_test(n, algo_opts, Δ)
    any(isnan.(n)) && return false
    @unpack c_delta, c_mu, mu = algo_opts
    return compatibility_test(n, c_delta, c_mu, mu, Δ)
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

function gridded_mop(
	mop :: AbstractMOP,
	lb = nothing,
	ub = nothing;
	res = 20
)
	lb = scatter_mop_lb(mop, lb)	
	ub = scatter_mop_ub(mop, ub)
	@assert all( ub .>= lb ) "Lower bounds exceed upper bounds."

	n = dim_vars(mop)
	@assert length(lb) == length(ub) == n 
	
	if length(res) == 1 && n > 1
		res = fill(only(res), n)
	end

	ax_args = [LinRange(lb[i], ub[i], res[i]) for i=1:n]
	tmp = init_value_caches(mop)
	fx = cached_fx(tmp)

	objf = (arg_vec) -> begin 
		_fx = similar(fx)
		objectives!(_fx, mop, arg_vec)
		return _fx
	end
	points = mapreduce(collect, hcat, Iterators.product(ax_args...))
	_Z = [objf(arg_vec) for arg_vec in eachcol(points)]
	K = dim_objectives(mop)
	Z = [[z[l] for z in _Z] for l=1:K]
	return ax_args, points, Z
end

function scatter_mop_lb(mop, lb)
	if isnothing(lb)
		lb = lower_var_bounds(mop)
	end
	if isnothing(lb)
		error("Provide non-nothing lower variable bounds `lb`.")
	end
	if any(isinf.(lb))
		error("Provide finite lower variable bounds `lb`.")
	end
	return lb
end
function scatter_mop_ub(mop, ub)
	if isnothing(ub)
		ub = upper_var_bounds(mop)
	end
	if isnothing(ub)
		error("Provide non-nothing upper variable bounds `ub`.")
	end
	if any(isinf.(ub))
		error("Provide finite upper variable bounds `ub`.")
	end
	return ub
end