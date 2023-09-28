# Stopping Criteria

Currently, the stopping behavior is entirely determined by certain fields
of the `AlgorithmOptions` object provided to `optimize` with the 
keyword argument `algo_opts`.

The relevant fields are 
* `max_iter`
* `stop_delta_min`, stop if the trust region radius is reduced to below `stop_delta_min`
* `stop_xtol_rel`, stop if the trial point ``xₜ`` is accepted and ``‖xₜ - x‖≤ δ‖x‖``.
* `stop_xtol_abs`, stop if the trial point ``xₜ`` is accepted and ``‖xₜ - x‖≤ ε``.
* `stop_ftol_rel`, stop if the trial point ``xₜ`` is accepted and ``‖f(xₜ) - f(x)‖≤ δ‖f(x)‖``
* `stop_ftol_abs`, stop if the trial point ``xₜ`` is accepted and ``‖f(xₜ) - f(x)‖≤ ε``.
* `stop_crit_tol_abs`, stop if for the approximate criticality it holds that ``χ̂(x) <= ε`` and for the feasibility that ``θ <= δ``.
* `stop_theta_tol_abs`, stop if for the approximate criticality it holds that ``χ̂(x) <= ε`` and for the feasibility that ``θ <= δ``.
* `stop_max_crit_loops`, stop after the criticality routine has looped `stop_max_crit_loops` times.

All values can be provided as keyword arguments to the `AlgorithmOptions`
constructor.
For defaults, instantiate an object without arguments,
```julia
algo_opts = AlgorithmOptions()
algo_opts.max_iter
```
or take a look at the source code.

## Internal
Internally, these options are converted to a tuple of `AbstractStoppingCriterion`s.
They implement the undocumented interface in `src/stopping.jl`.
A stopping criterion can be checked at different positions in the loop,
indicated by methods 
`check_pre_iteration`, `check_post_descent_step`, `check_post_iteration`,
`check_pre_crit_loop` and `check_post_crit_loop`.
At the relevant position, the corresponding method
```julia
function evaluate_stopping_criterion(
    crit::AbstractStoppingCriterion,
    Δ, mop, mod, scaler, lin_cons, scaled_cons,
    vals, vals_tmp, step_vals, mod_vals, filter, iter_meta, step_cache, algo_opts,
)
    return nothing
end
```
is called.
If it returns something other than `nothing`, then the algorithm stops.
`iter_meta` already stores radius information, but not for the criticality
loop. 
That is why it can be provided as its own argument `Δ` (at least for now).
The contents of the arguments differ depending on the position of evaluation within the optimization loop.

## User Callbacks
You can implement the same interface for your own callback to
gather information and also stop the algorithm based on your 
own criteria.
```julia
Base.@kwdef struct MyGatheringCallback  <: Compromise.AbstractStoppingCriterion 
    iterates :: Matrix{Float64} = Matrix{Float64}(undef, 2, 2) 
    it_counter :: Base.RefValue{Int} = Ref(0)
end
Compromise.check_pre_iteration(::MyGatheringCallback)=true
        
function Compromise.evaluate_stopping_criterion(
    crit::MyGatheringCallback,
    Δ, mop, mod, scaler, lin_cons, scaled_cons,
    vals, vals_tmp, step_vals, mod_vals, filter, iter_meta, step_cache, algo_opts,
)
    i = crit.it_counter[] += 1
    i > 2 && return crit
    crit.iterates[:,i] .= vals.x
    return nothing
end

mop = MutableMOP(;num_vars=2)
add_objectives!(
    mop, x -> [sum((x.-1).^2), sum((x.+1).^2)], :rbf; 
    dim_out=2, func_iip=false,
)
final_vals, stop_code = optimize(mop, rand(2); user_callback=MyGatheringCallback())
```