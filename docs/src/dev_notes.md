# # Developer Notes

# ## Function Signatures

For utility functions with few arguments, we try to keep to the Julia
convention of appending an exclamation mark, if a method modifies some 
of its arguments, and place those arguments first.

However, some more complicated functions essential to the algorithm, 
like `do_iteration`, `do_restoration`, `compute_normal_step` and `compute_descent_step`, 
can require a multitude of arguments and keeping the correct order during development proves difficult.
That is why in this case we use the standard signature
```julia
function algo_function(
    # iteration information
    it_index, Δ, it_stat,
    # objects to evaluate objectives, models and constraints
    mop, mod, scaler, lin_cons, scaled_cons,
    # caches for necessary arrays
    vals, vals_tmp, step_vals, mod_vals, 
    # other important building blocks
    filter,     # the filter used to drive feasibility
    step_cache, # an object defining step calculation and holding caches
    algo_opts;  # general algorithmic settings
    # then follow custom keyword arguments …
    custom_kwarg1, kwargs...
)
    # Function Body
end
```
We do not use keyword arguments to enable dispatch on custom configuration
or cache types, e.g., in the case of `compute_descent_step`.
What arguments are modified, or even guaranteed or required to be modified, should be made clear from docstrings or comments.
The compiler does only specialize on function arguments if they are used within in 
the function body, but not if they are merely passed through to other functions.

# ## Medium-Priority ToDo's

* Introduce `AbstractSurrogateModelConfig` to initialize `AbstractSurrogateModel` based
  on custom configuration objects instead of their types.
* Make `dim_in` or `num_vars` part of the `AbstractMOP` and `AbstractMOPSurrogate` interface.
* Add dimension information to `AbstractNonlinearOperator` and `AbstractSurrogateModel`.  
  When this is done, some method signatures concerning initialization and updates can be  
  simplified. (`CTRL-F` for “dim_in”, “num_vars”, “num_out”, “dim_out” etc.)
* In the modelling and DAG framework, switch names and meaning of dependent variables and  
  states. 



