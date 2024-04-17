struct ReturnObject{X, C, S}
	ξ0 :: X
	cache :: C
    stop_code :: S
end

opt_initial_vars(r::ReturnObject) = r.ξ0
opt_cache(r::ReturnObject) = r.cache
opt_vals(r::ReturnObject) = opt_cache(r).vals
opt_surrogate(r::ReturnObject) = opt_cache(r).mod

opt_vars(r::ReturnObject)=opt_vars(opt_vals(r))
opt_vars(::Nothing) = missing
opt_vars(v::AbstractMOPCache)=cached_ξ(v)

opt_objectives(r::ReturnObject)=opt_objectives(opt_vals(r))
opt_objectives(::Nothing)=missing
opt_objectives(v::AbstractMOPCache)=cached_fx(v)

opt_nl_eq_constraints(r::ReturnObject)=opt_nl_eq_constraints(opt_vals(r))
opt_nl_eq_constraints(::Nothing)=missing
opt_nl_eq_constraints(v::AbstractMOPCache)=cached_hx(v)

opt_nl_ineq_constraints(r::ReturnObject)=opt_nl_ineq_constraints(opt_vals(r))
opt_nl_ineq_constraints(::Nothing)=missing
opt_nl_ineq_constraints(v::AbstractMOPCache)=cached_gx(v)

opt_lin_eq_constraints(r::ReturnObject)=opt_lin_eq_constraints(opt_vals(r))
opt_lin_eq_constraints(::Nothing)=missing
opt_lin_eq_constraints(v::AbstractMOPCache)=cached_Ex_min_c(v)

opt_lin_ineq_constraints(r::ReturnObject)=opt_lin_ineq_constraints(opt_vals(r))
opt_lin_ineq_constraints(::Nothing)=missing
opt_lin_ineq_constraints(v::AbstractMOPCache)=cached_Ax_min_b(v)

opt_constraint_violation(r::ReturnObject)=opt_constraint_violation(opt_vals(r))
opt_constraint_violation(::Nothing)=missing
opt_constraint_violation(v::AbstractMOPCache)=cached_theta(v)

function opt_stop_code(r::ReturnObject)
    return unwrap_stop_crit(r.stop_code)
end

function Base.show(io::IO, ret::ReturnObject)
	print(io, """
	ReturnObject
	x0   = $(pretty_row_vec(opt_initial_vars(ret)))
	x*   = $(pretty_row_vec(opt_vars(ret)))
	f(x*)= $(pretty_row_vec(opt_objectives(ret)))
	code = $(opt_stop_code(ret))"""
	)
end