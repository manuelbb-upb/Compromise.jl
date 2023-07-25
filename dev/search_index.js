var documenterSearchIndex = {"docs":
[{"location":"mop/#AbstractMOP-Interface","page":"AbstractMOP Interface","title":"AbstractMOP Interface","text":"","category":"section"},{"location":"mop/","page":"AbstractMOP Interface","title":"AbstractMOP Interface","text":"An object subtyping AbstractMOP is a glorified wrapper around vector-vector functions. The methods below were originally meant to be used to implement our algorithm similar to how it has been stated in the article, rather “mathematically“. That is, we do not care for how the problem has been modelled and set up. We only need function handles and some meta-data concerning dimensions and data types.","category":"page"},{"location":"mop/","page":"AbstractMOP Interface","title":"AbstractMOP Interface","text":"Formally, our problem reads","category":"page"},{"location":"mop/","page":"AbstractMOP Interface","title":"AbstractMOP Interface","text":"beginaligned\nmin_x = x₁  x_N\n    beginbmatrix\n        f₁(x)   \n               \n        f_K(x)\n    endbmatrix\n        textsubject to\n     g(x) = g₁(x)  g_P(x)  0 \n     h(x) = h₁(x)  h_M(x) = 0 \n     lb  x  ub  A  x  b E  x = c\nendaligned","category":"page"},{"location":"mop/","page":"AbstractMOP Interface","title":"AbstractMOP Interface","text":"In the code, we often follow this notation:","category":"page"},{"location":"mop/","page":"AbstractMOP Interface","title":"AbstractMOP Interface","text":"f indicates an objective function.\ng a nonlinear inequality constraint.\nh a nonlinear equality constriant.\nA is the matrix of linear inequality constraints, b the right hand side vector.\nE is the matrix of linear equality constraints, c the right hand side vector.\nlb and ub define box constraints.","category":"page"},{"location":"mop/","page":"AbstractMOP Interface","title":"AbstractMOP Interface","text":"At the beginning of an optimization routine, initialization based on the initial site can be performed:","category":"page"},{"location":"mop/","page":"AbstractMOP Interface","title":"AbstractMOP Interface","text":"initialize(mop::AbstractMOP, ξ0::RVec)=mop","category":"page"},{"location":"mop/#Meta-Data","page":"AbstractMOP Interface","title":"Meta-Data","text":"","category":"section"},{"location":"mop/","page":"AbstractMOP Interface","title":"AbstractMOP Interface","text":"The optional function precision returns the type of result and derivative vectors:","category":"page"},{"location":"mop/","page":"AbstractMOP Interface","title":"AbstractMOP Interface","text":"precision(::AbstractMOP)::Type{<:AbstractFloat}=DEFAULT_PRECISION","category":"page"},{"location":"mop/","page":"AbstractMOP Interface","title":"AbstractMOP Interface","text":"We would also like to deterministically query the expected surrogate model types:","category":"page"},{"location":"mop/","page":"AbstractMOP Interface","title":"AbstractMOP Interface","text":"model_type(::AbstractMOP)::Type{<:AbstractMOPSurrogate}=AbstractMOPSurrogate","category":"page"},{"location":"mop/","page":"AbstractMOP Interface","title":"AbstractMOP Interface","text":"Below functions are used to query dimension information.","category":"page"},{"location":"mop/","page":"AbstractMOP Interface","title":"AbstractMOP Interface","text":"dim_objectives(::AbstractMOP)::Int=0            # mandatory\ndim_nl_eq_constraints(::AbstractMOP)::Int=0     # optional\ndim_nl_ineq_constraints(::AbstractMOP)::Int=0   # optional","category":"page"},{"location":"mop/#Linear-Constraints","page":"AbstractMOP Interface","title":"Linear Constraints","text":"","category":"section"},{"location":"mop/","page":"AbstractMOP Interface","title":"AbstractMOP Interface","text":"An AbstractMOP can have constrained variables. The corresponding functions should return full bound vectors or nothing. For lower bounds, nothing corresponds to -Inf, but we do not necessarily use such vectors in the inner solver. Upper bounds would be Inf in case of nothing.","category":"page"},{"location":"mop/","page":"AbstractMOP Interface","title":"AbstractMOP Interface","text":"lower_var_bounds(::AbstractMOP)::Union{Nothing, Vec}=nothing\nupper_var_bounds(::AbstractMOP)::Union{Nothing, Vec}=nothing","category":"page"},{"location":"mop/","page":"AbstractMOP Interface","title":"AbstractMOP Interface","text":"Moreover, problems can have linear equality constraints and linear inequality constraints","category":"page"},{"location":"mop/","page":"AbstractMOP Interface","title":"AbstractMOP Interface","text":"  E x = c\n  quad\n  A x  b","category":"page"},{"location":"mop/","page":"AbstractMOP Interface","title":"AbstractMOP Interface","text":"lin_eq_constraints(::AbstractMOP)::Union{Nothing, Tuple{RMat,RVecOrMat}}=nothing\nlin_ineq_constraints(::AbstractMOP)::Union{Nothing, Tuple{RMat,RVecOrMat}}=nothing","category":"page"},{"location":"mop/","page":"AbstractMOP Interface","title":"AbstractMOP Interface","text":"From that we can derive dimension getters as well:","category":"page"},{"location":"mop/","page":"AbstractMOP Interface","title":"AbstractMOP Interface","text":"# helper\ndim_lin_constraints(dat::Nothing)=0\nfunction dim_lin_constraints((A,b)::Tuple{RMat, RVecOrMat})\n    dim = length(b)\n    @assert size(A, 2) == dim \"Dimension mismatch in linear constraints.\"\n    return dim\nend\n# actual functions\ndim_lin_eq_constraints(mop::AbstractMOP)=dim_lin_constraints(lin_eq_constraints(mop))\ndim_lin_ineq_constraints(mop::AbstractMOP)=dim_lin_constraints(lin_ineq_constraints(mop))","category":"page"},{"location":"mop/#Evaluation","page":"AbstractMOP Interface","title":"Evaluation","text":"","category":"section"},{"location":"mop/","page":"AbstractMOP Interface","title":"AbstractMOP Interface","text":"Evaluation of nonlinear objective functions requires the following method:","category":"page"},{"location":"mop/","page":"AbstractMOP Interface","title":"AbstractMOP Interface","text":"function eval_objectives!(y::RVec, mop::M, x::RVec) where {M<:AbstractMOP}\n    error(\"`eval_objectives!(y, mop, x) not implemented for mop of type $(M).\")\nend","category":"page"},{"location":"mop/","page":"AbstractMOP Interface","title":"AbstractMOP Interface","text":"If there are constraints, these have to be defined as well:","category":"page"},{"location":"mop/","page":"AbstractMOP Interface","title":"AbstractMOP Interface","text":"function eval_nl_eq_constraints!(y::RVec, mop::M, x::RVec) where {M<:AbstractMOP}\n    error(\"`eval_nl_eq_constraints!(y, mop, x) not implemented for mop of type $(M).\")\nend\nfunction eval_nl_ineq_constraints!(y::RVec, mop::M, x::RVec) where {M<:AbstractMOP}\n    error(\"`eval_nl_ineq_constraints!(y, mop, x) not implemented for mop of type $(M).\")\nend","category":"page"},{"location":"mop/","page":"AbstractMOP Interface","title":"AbstractMOP Interface","text":"To ensure, they only get called if needed, we wrap them and assign shorter names:","category":"page"},{"location":"mop/","page":"AbstractMOP Interface","title":"AbstractMOP Interface","text":"objectives!(y::RVec, mop::AbstractMOP, x::RVec)=eval_objectives!(y, mop, x)\nnl_eq_constraints!(y::Nothing, mop::AbstractMOP, x::RVec)=nothing\nnl_ineq_constraints!(y::Nothing, mop::AbstractMOP, x::RVec)=nothing\nnl_eq_constraints!(y::RVec, mop::AbstractMOP, x::RVec)=eval_nl_eq_constraints!(y, mop, x)\nnl_ineq_constraints!(y::RVec, mop::AbstractMOP, x::RVec)=eval_nl_ineq_constraints!(y, mop, x)","category":"page"},{"location":"mop/","page":"AbstractMOP Interface","title":"AbstractMOP Interface","text":"Similar methods can be defined for linear constraints:","category":"page"},{"location":"mop/","page":"AbstractMOP Interface","title":"AbstractMOP Interface","text":"lin_cons!(y::Nothing, cons::Nothing, x::RVec) = nothing\nfunction lin_cons!(y::RVec, (A, b)::Tuple, x::RVec)\n    LA.mul!(y, A, x)\n    y .-= b\n    return nothing\nend\nlin_eq_constraints!(y::Nothing, mop::AbstractMOP, x::RVec)=nothing\nlin_ineq_constraints!(y::Nothing, mop::AbstractMOP, x::RVec)=nothing\nlin_eq_constraints!(y::RVec, mop::AbstractMOP, x::RVec)=lin_cons!(y, lin_eq_constraints(mop), x)\nlin_ineq_constraints!(y::RVec, mop::AbstractMOP, x::RVec)=lin_cons!(y, lin_ineq_constraints(mop), x)","category":"page"},{"location":"mop/#Pre-Allocation","page":"AbstractMOP Interface","title":"Pre-Allocation","text":"","category":"section"},{"location":"mop/","page":"AbstractMOP Interface","title":"AbstractMOP Interface","text":"Why do we also allow nothing as the target for constraints? Because that is the default cache returned if there are none:","category":"page"},{"location":"mop/","page":"AbstractMOP Interface","title":"AbstractMOP Interface","text":"function prealloc_objectives_vector(mop::AbstractMOP)\n    T = precision(mop)\n    return Vector{T}(undef, dim_objectives(mop))\nend\n# hx = nonlinear equality constraints at x\n# gx = nonlinear inequality constraints at x\n# Ex = linear equality constraints at x\n# Ax = linear inequality constraints at x\n# These are defined below (and I put un-specific definitions here for the Linter)\nfunction prealloc_nl_eq_constraints_vector(mop) end\nfunction prealloc_nl_ineq_constraints_vector(mop) end\nfunction prealloc_lin_eq_constraints_vector(mop) end\nfunction prealloc_lin_ineq_constraints_vector(mop) end\nfor (dim_func, prealloc_func) in (\n    (:dim_nl_eq_constraints, :prealloc_nl_eq_constraints_vector),\n    (:dim_nl_ineq_constraints, :prealloc_nl_ineq_constraints_vector),\n    (:dim_lin_eq_constraints, :prealloc_lin_eq_constraints_vector),\n    (:dim_lin_ineq_constraints, :prealloc_lin_ineq_constraints_vector),\n)\n    @eval function $(prealloc_func)(mop::AbstractMOP)\n        dim = $(dim_func)(mop)\n        if dim > 0\n            T = precision(mop)\n            return Vector{T}(undef, dim)\n        else\n            return nothing\n        end\n    end\nend","category":"page"},{"location":"mop/","page":"AbstractMOP Interface","title":"AbstractMOP Interface","text":"","category":"page"},{"location":"mop/","page":"AbstractMOP Interface","title":"AbstractMOP Interface","text":"This page was generated using Literate.jl.","category":"page"},{"location":"models/#AbstractMOPSurrogate-Interface","page":"AbstractMOPSurrogate Interface","title":"AbstractMOPSurrogate Interface","text":"","category":"section"},{"location":"models/","page":"AbstractMOPSurrogate Interface","title":"AbstractMOPSurrogate Interface","text":"The speciality of our algorithm is its use of local surrogate models. These should subtype and implement AbstractMOPSurrogate. Every nonlinear function can be modelled, but we leave it to the implementation of an AbstractMOP, how exactly that is done.","category":"page"},{"location":"models/#Meta-Data","page":"AbstractMOPSurrogate Interface","title":"Meta-Data","text":"","category":"section"},{"location":"models/","page":"AbstractMOPSurrogate Interface","title":"AbstractMOPSurrogate Interface","text":"For convenience, we'd like to have the same meta information available as for the original MOP:","category":"page"},{"location":"models/","page":"AbstractMOPSurrogate Interface","title":"AbstractMOPSurrogate Interface","text":"precision(::AbstractMOPSurrogate)::Type{<:AbstractFloat}=DEFAULT_PRECISION\ndim_objectives(::AbstractMOPSurrogate)::Int=0            # mandatory\ndim_nl_eq_constraints(::AbstractMOPSurrogate)::Int=0     # optional\ndim_nl_ineq_constraints(::AbstractMOPSurrogate)::Int=0   # optional","category":"page"},{"location":"models/","page":"AbstractMOPSurrogate Interface","title":"AbstractMOPSurrogate Interface","text":"Additionally, we require information on the model variability and if we can build models for the scaled domain:","category":"page"},{"location":"models/","page":"AbstractMOPSurrogate Interface","title":"AbstractMOPSurrogate Interface","text":"depends_on_trust_region(::AbstractMOPSurrogate)::Bool=true\nsupports_scaling(T::Type{<:AbstractMOPSurrogate})=NoScaling()","category":"page"},{"location":"models/#Construction","page":"AbstractMOPSurrogate Interface","title":"Construction","text":"","category":"section"},{"location":"models/","page":"AbstractMOPSurrogate Interface","title":"AbstractMOPSurrogate Interface","text":"Define a function to return a model for some MOP. The model does not yet have to be trained.","category":"page"},{"location":"models/","page":"AbstractMOPSurrogate Interface","title":"AbstractMOPSurrogate Interface","text":"init_models(mop::AbstractMOP, n_vars, scaler)::AbstractMOPSurrogate=nothing","category":"page"},{"location":"models/","page":"AbstractMOPSurrogate Interface","title":"AbstractMOPSurrogate Interface","text":"It is trained with the update method.","category":"page"},{"location":"models/","page":"AbstractMOPSurrogate Interface","title":"AbstractMOPSurrogate Interface","text":"function update_models!(\n    mod::AbstractMOPSurrogate, Δ, mop, scaler, vals, scaled_cons, algo_opts; point_has_changed\n)\n    return nothing\nend","category":"page"},{"location":"models/#Evaluation","page":"AbstractMOPSurrogate Interface","title":"Evaluation","text":"","category":"section"},{"location":"models/","page":"AbstractMOPSurrogate Interface","title":"AbstractMOPSurrogate Interface","text":"Evaluation of nonlinear objective models requires the following method. x will be from the scaled domain, but if a model does not support scaling, then internally the IdentityScaler() is used:","category":"page"},{"location":"models/","page":"AbstractMOPSurrogate Interface","title":"AbstractMOPSurrogate Interface","text":"function eval_objectives!(y::RVec, mod::M, x::RVec) where {M<:AbstractMOPSurrogate}\n    error(\"`eval_objectives!(y, mod, x) not implemented for mod of type $(M).\")\nend","category":"page"},{"location":"models/","page":"AbstractMOPSurrogate Interface","title":"AbstractMOPSurrogate Interface","text":"If there are constraints, these have to be defined as well:","category":"page"},{"location":"models/","page":"AbstractMOPSurrogate Interface","title":"AbstractMOPSurrogate Interface","text":"function eval_nl_eq_constraints!(y::RVec, mod::M, x::RVec) where {M<:AbstractMOPSurrogate}\n    error(\"`eval_nl_eq_constraints!(y, mod, x) not implemented for mod of type $(M).\")\nend\nfunction eval_nl_ineq_constraints!(y::RVec, mod::M, x::RVec) where {M<:AbstractMOPSurrogate}\n    error(\"`eval_nl_ineq_constraints!(y, mod, x) not implemented for mod of type $(M).\")\nend","category":"page"},{"location":"models/","page":"AbstractMOPSurrogate Interface","title":"AbstractMOPSurrogate Interface","text":"As before, we use shorter function names in the algorithm.","category":"page"},{"location":"models/","page":"AbstractMOPSurrogate Interface","title":"AbstractMOPSurrogate Interface","text":"objectives!(y::RVec, mod::AbstractMOPSurrogate, x::RVec)=eval_objectives!(y, mod, x)\nnl_eq_constraints!(y::Nothing, mod::AbstractMOPSurrogate, x::RVec)=nothing\nnl_ineq_constraints!(y::Nothing, mod::AbstractMOPSurrogate, x::RVec)=nothing\nnl_eq_constraints!(y::RVec, mod::AbstractMOPSurrogate, x::RVec)=eval_nl_eq_constraints!(y, mod, x)\nnl_ineq_constraints!(y::RVec, mod::AbstractMOPSurrogate, x::RVec)=eval_nl_ineq_constraints!(y, mod, x)","category":"page"},{"location":"models/#Pre-Allocation","page":"AbstractMOPSurrogate Interface","title":"Pre-Allocation","text":"","category":"section"},{"location":"models/","page":"AbstractMOPSurrogate Interface","title":"AbstractMOPSurrogate Interface","text":"The preallocation functions look the same as for AbstractMOP:","category":"page"},{"location":"models/","page":"AbstractMOPSurrogate Interface","title":"AbstractMOPSurrogate Interface","text":"for (dim_func, prealloc_func) in (\n    (:dim_objectives, :prealloc_objectives_vector),\n    (:dim_nl_eq_constraints, :prealloc_nl_eq_constraints_vector),\n    (:dim_nl_ineq_constraints, :prealloc_nl_ineq_constraints_vector),\n)\n    @eval function $(prealloc_func)(mod::AbstractMOPSurrogate)\n        dim = $(dim_func)(mod)\n        if dim > 0\n            T = precision(mod)\n            return Vector{T}(undef, dim)\n        else\n            return nothing\n        end\n    end\nend","category":"page"},{"location":"models/#Differentiation","page":"AbstractMOPSurrogate Interface","title":"Differentiation","text":"","category":"section"},{"location":"models/","page":"AbstractMOPSurrogate Interface","title":"AbstractMOPSurrogate Interface","text":"The surrogate models are also used to query approximate derivative information. We hence need the following functions to make Dy transposed model Jacobians:","category":"page"},{"location":"models/","page":"AbstractMOPSurrogate Interface","title":"AbstractMOPSurrogate Interface","text":"function grads_objectives!(Dy::RMat, mod::M, x::RVec) where {M<:AbstractMOPSurrogate}\n    error(\"`grads_objectives!(Dy, mod, x) not implemented for mod of type $(M).\")\nend","category":"page"},{"location":"models/","page":"AbstractMOPSurrogate Interface","title":"AbstractMOPSurrogate Interface","text":"If there are constraints, these have to be defined as well:","category":"page"},{"location":"models/","page":"AbstractMOPSurrogate Interface","title":"AbstractMOPSurrogate Interface","text":"function grads_nl_eq_constraints!(Dy::RMat, mod::M, x::RVec) where {M<:AbstractMOPSurrogate}\n    error(\"`grads_nl_eq_constraints!(Dy, mod, x) not implemented for mod of type $(M).\")\nend\nfunction grads_nl_ineq_constraints!(Dy::RMat, mod::M, x::RVec) where {M<:AbstractMOPSurrogate}\n    error(\"`grads_nl_ineq_constraints!(Dy, mod, x) not implemented for mod of type $(M).\")\nend","category":"page"},{"location":"models/","page":"AbstractMOPSurrogate Interface","title":"AbstractMOPSurrogate Interface","text":"Here, the names of the wrapper functions start with “diff“.","category":"page"},{"location":"models/","page":"AbstractMOPSurrogate Interface","title":"AbstractMOPSurrogate Interface","text":"diff_objectives!(Dy::RMat, mod::AbstractMOPSurrogate, x::RVec)=grads_objectives!(Dy, mod, x)\ndiff_nl_eq_constraints!(Dy::Nothing, mod::AbstractMOPSurrogate, x::RVec)=nothing\ndiff_nl_ineq_constraints!(Dy::Nothing, mod::AbstractMOPSurrogate, x::RVec)=nothing\ndiff_nl_eq_constraints!(Dy::RMat, mod::AbstractMOPSurrogate, x::RVec)=grads_nl_eq_constraints!(Dy, mod, x)\ndiff_nl_ineq_constraints!(Dy::RMat, mod::AbstractMOPSurrogate, x::RVec)=grads_nl_ineq_constraints!(Dy, mod, x)","category":"page"},{"location":"models/#Optionally,-we-can-have-evaluation-and-differentiation-in-one-go:","page":"AbstractMOPSurrogate Interface","title":"Optionally, we can have evaluation and differentiation in one go:","text":"","category":"section"},{"location":"models/","page":"AbstractMOPSurrogate Interface","title":"AbstractMOPSurrogate Interface","text":"function eval_and_grads_objectives!(y::RVec, Dy::RMat, mod::M, x::RVec) where {M<:AbstractMOPSurrogate}\n    eval_objectives!(y, mop, x)\n    grads_objectives!(Dy, mod, x)\n    return nothing\nend\nfunction eval_grads_nl_eq_constraints!(y::RVec, Dy::RMat, mod::M, x::RVec) where {M<:AbstractMOPSurrogate}\n    eval_nl_eq_constraints!(y, mop, x)\n    grads_nl_eq_constraints!(Dy, mod, x)\n    return nothing\nend\nfunction eval_grads_nl_ineq_constraints!(y::RVec, Dy::RMat, mod::M, x::RVec) where {M<:AbstractMOPSurrogate}\n    eval_nl_ineq_constraints!(y, mop, x)\n    grads_nl_ineq_constraints!(Dy, mod, x)\n    return nothing\nend","category":"page"},{"location":"models/","page":"AbstractMOPSurrogate Interface","title":"AbstractMOPSurrogate Interface","text":"Wrappers for use in the algorithm:","category":"page"},{"location":"models/","page":"AbstractMOPSurrogate Interface","title":"AbstractMOPSurrogate Interface","text":"vals_diff_objectives!(y::RVec, Dy::RMat, mod::AbstractMOPSurrogate, x::RVec)=eval_and_grads_objectives!(y, Dy, mod, x)\nvals_diff_nl_eq_constraints!(y::Nothing, Dy::Nothing, mod::AbstractMOPSurrogate, x::RVec)=nothing\nvals_diff_nl_ineq_constraints!(y::Nothing, Dy::Nothing, mod::AbstractMOPSurrogate, x::RVec)=nothing\nvals_diff_nl_eq_constraints!(y::RVec, Dy::RMat, mod::AbstractMOPSurrogate, x::RVec)=eval_and_grads_nl_eq_constraints!(y, Dy, mod, x)\nvals_diff_nl_ineq_constraints!(y::RVec, Dy::RMat, mod::AbstractMOPSurrogate, x::RVec)=eval_and_grads_nl_ineq_constraints!(y, Dy, mod, x)","category":"page"},{"location":"models/","page":"AbstractMOPSurrogate Interface","title":"AbstractMOPSurrogate Interface","text":"Here is what is called later on:","category":"page"},{"location":"models/","page":"AbstractMOPSurrogate Interface","title":"AbstractMOPSurrogate Interface","text":"\"Evaluate the models `mod` at `x` and store results in `mod_vals::SurrogateValueArrays`.\"\nfunction eval_mod!(mod_vals, mod, x)\n    @unpack fx, hx, gx = mod_vals\n    objectives!(fx, mod, x)\n    nl_eq_constraints!(hx, mod, x)\n    nl_ineq_constraints!(hx, mod, x)\n    return nothing\nend\n\n\"Evaluate the model gradients of `mod` at `x` and store results in `mod_vals::SurrogateValueArrays`.\"\nfunction diff_mod!(mod_vals, mod, x)\n    @unpack Dfx, Dhx, Dgx = mod_vals\n    diff_objectives!(Dfx, mod, x)\n    diff_nl_eq_constraints!(Dhx, mod, x)\n    diff_nl_ineq_constraints!(hx, mod, x)\n    return nothing\nend\n\n\"Evaluate and differentiate `mod` at `x` and store results in `mod_vals::SurrogateValueArrays`.\"\nfunction eval_and_diff_mod!(mod_vals, mod, x)\n    @unpack fx, hx, gx, Dfx, Dhx, Dgx = mod_vals\n    vals_diff_objectives!(fx, Dfx, mod, x)\n    vals_diff_nl_eq_constraints!(hx, Dhx, mod, x)\n    vals_diff_nl_ineq_constraints!(gx, Dgx, mod, x)\n    return nothing\nend","category":"page"},{"location":"models/#Gradient-Pre-Allocation","page":"AbstractMOPSurrogate Interface","title":"Gradient Pre-Allocation","text":"","category":"section"},{"location":"models/","page":"AbstractMOPSurrogate Interface","title":"AbstractMOPSurrogate Interface","text":"We also would like to have pre-allocated gradient arrays ready:","category":"page"},{"location":"models/","page":"AbstractMOPSurrogate Interface","title":"AbstractMOPSurrogate Interface","text":"function prealloc_objectives_grads(mod::AbstractMOPSurrogate, n_vars)\n    T = precision(mod)\n    return Matrix{T}(undef, n_vars, dim_objectives(mod))\nend\n# These are defined below (and I put un-specific definitions here for the Linter)\nfunction prealloc_nl_eq_constraints_grads(mod, n_vars) end\nfunction prealloc_nl_ineq_constraints_grads(mod, n_vars) end\nfor (dim_func, prealloc_func) in (\n    (:dim_nl_eq_constraints, :prealloc_nl_eq_constraints_grads),\n    (:dim_nl_ineq_constraints, :prealloc_nl_ineq_constraints_grads),\n)\n    @eval function $(prealloc_func)(mod::AbstractMOPSurrogate, n_vars)\n        n_out = $(dim_func)(mod)\n        if n_out > 0\n            T = precision(mod)\n            return Matrix{T}(undef, n_vars, n_out)\n        else\n            return nothing\n        end\n    end\nend","category":"page"},{"location":"models/","page":"AbstractMOPSurrogate Interface","title":"AbstractMOPSurrogate Interface","text":"note: Note\nFor nonlinear subproblem solvers it might be desirable to have partial evaluation and differentiation functions. Also, out-of-place functions could be useful for external nonlinear tools, but I don't need them yet. Defining the latter methods would simply call prealloc_XXX first and then use some in-place-functions.","category":"page"},{"location":"models/","page":"AbstractMOPSurrogate Interface","title":"AbstractMOPSurrogate Interface","text":"","category":"page"},{"location":"models/","page":"AbstractMOPSurrogate Interface","title":"AbstractMOPSurrogate Interface","text":"This page was generated using Literate.jl.","category":"page"},{"location":"dev_notes/#Developer-Notes","page":"(Dev) Notes","title":"Developer Notes","text":"","category":"section"},{"location":"dev_notes/#Function-Signatures","page":"(Dev) Notes","title":"Function Signatures","text":"","category":"section"},{"location":"dev_notes/","page":"(Dev) Notes","title":"(Dev) Notes","text":"For utility functions with few arguments, we try to keep to the Julia convention of appending an exclamation mark, if a method modifies some  of its arguments, and place those arguments first.","category":"page"},{"location":"dev_notes/","page":"(Dev) Notes","title":"(Dev) Notes","text":"However, some more complicated functions essential to the algorithm,  like do_iteration, do_restoration, compute_normal_step and compute_descent_step,  can require a multitude of arguments and keeping the correct order during development proves difficult. That is why in this case we use the standard signature","category":"page"},{"location":"dev_notes/","page":"(Dev) Notes","title":"(Dev) Notes","text":"function algo_function(\n    # iteration information\n    it_index, Δ, it_stat,\n    # objects to evaluate objectives, models and constraints\n    mop, mod, scaler, lin_cons, scaled_cons,\n    # caches for necessary arrays\n    vals, vals_tmp, step_vals, mod_vals, \n    # other important building blocks\n    filter,     # the filter used to drive feasibility\n    step_cache, # an object defining step calculation and holding caches\n    algo_opts;  # general algorithmic settings\n    # then follow custom keyword arguments …\n    custom_kwarg1, kwargs...\n)\n    # Function Body\nend","category":"page"},{"location":"dev_notes/","page":"(Dev) Notes","title":"(Dev) Notes","text":"We do not use keyword arguments to enable dispatch on custom configuration or cache types, e.g., in the case of compute_descent_step. What arguments are modified, or even guaranteed or required to be modified, should be made clear from docstrings or comments. The compiler does only specialize on function arguments if they are used within in  the function body, but not if they are merely passed through to other functions.","category":"page"},{"location":"dev_notes/#Medium-Priority-To-Do's","page":"(Dev) Notes","title":"Medium-Priority To-Do's","text":"","category":"section"},{"location":"dev_notes/","page":"(Dev) Notes","title":"(Dev) Notes","text":"Introduce AbstractSurrogateModelConfig to initialize AbstractSurrogateModel based on custom configuration objects instead of their types.\nMake dim_in or num_vars part of the AbstractMOP and AbstractMOPSurrogate interface.\nAdd dimension information to AbstractNonlinearOperator and AbstractSurrogateModel.   When this is done, some method signatures concerning initialization and updates can be   simplified. (CTRL-F for dim_in, num_vars, num_out, dim_out etc.)\nIn the modelling and DAG framework, switch names and meaning of dependent variables and   states. ","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = Compromise","category":"page"},{"location":"#Compromise","page":"Home","title":"Compromise","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for Compromise. There is not much here yet. Everything is still very much a work-in-progress.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Random Doc-Strings:","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [Compromise]","category":"page"},{"location":"#Compromise.compute_descent_step-Union{Tuple{SC}, Tuple{Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, SC, Any}} where SC<:Compromise.AbstractStepCache","page":"Home","title":"Compromise.compute_descent_step","text":"compute_descent_step(\n    it_index, Δ, it_stat, mop, mod, scaler, lin_cons, scaled_cons,\n    vals, vals_tmp, step_vals, mod_vals, filter, step_cache, algo_opts\n) :: Real\n\nReturn a criticalty value χ.\nModify step_vals.d to hold the scaled descent step at step_vals.xn.\nModify step_vals.xs to hold vals.x + step_vals.n + step_vals.d,  or equivalently, step_vals.xn + step_vals.d.\nModify step_vals.fxs to hold the surrogate objective values at step_vals.xs.\nThe descent step is computed according to step_cache and this cache can also be  modified.\nUsually, the descent step is computed using the surrogate values stored in mod_vals.   These should not be modified!\n\n\n\n\n\n","category":"method"},{"location":"#Compromise.compute_normal_step-Union{Tuple{SC}, Tuple{Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, SC, Any}} where SC<:Compromise.AbstractStepCache","page":"Home","title":"Compromise.compute_normal_step","text":"compute_normal_step(\n    it_index, Δ, it_stat, mop, mod, scaler, lin_cons, scaled_cons,\n    vals, vals_tmp, step_vals, mod_vals, filter, step_cache, algo_opts\n)\n\nModify step_vals.n to hold the normal step at vals.x.\nModify step_vals.xn to hold vals.x + step_vals.n.\nThe normal step is computed according to step_cache and this cache can also be  modified.\nUsually, the normal step is computed using the surrogate values stored in mod_vals.   These should not be modified!\n\n\n\n\n\n","category":"method"},{"location":"#Compromise.diff_mod!-Tuple{Any, Any, Any}","page":"Home","title":"Compromise.diff_mod!","text":"Evaluate the model gradients of mod at x and store results in mod_vals::SurrogateValueArrays.\n\n\n\n\n\n","category":"method"},{"location":"#Compromise.eval_and_diff_mod!-Tuple{Any, Any, Any}","page":"Home","title":"Compromise.eval_and_diff_mod!","text":"Evaluate and differentiate mod at x and store results in mod_vals::SurrogateValueArrays.\n\n\n\n\n\n","category":"method"},{"location":"#Compromise.eval_mod!-Tuple{Any, Any, Any}","page":"Home","title":"Compromise.eval_mod!","text":"Evaluate the models mod at x and store results in mod_vals::SurrogateValueArrays.\n\n\n\n\n\n","category":"method"},{"location":"#Compromise.scale!-Tuple{Any, Compromise.AbstractAffineScaler, Any}","page":"Home","title":"Compromise.scale!","text":"Scale ξ and set x according to x = T*ξ + t.\n\n\n\n\n\n","category":"method"},{"location":"#Compromise.scale_eq!-Tuple{Any, Compromise.AbstractAffineScaler, Any}","page":"Home","title":"Compromise.scale_eq!","text":"Make Aξ + b ? 0 applicable in scaled domain via A(inv(T)*x - inv(T)*t) + b ? 0.\n\n\n\n\n\n","category":"method"},{"location":"#Compromise.set_linear_constraints!-Tuple{Any, Any, Any, Any, Any, Symbol}","page":"Home","title":"Compromise.set_linear_constraints!","text":"Add the constraint c + A * x .?= b to opt and return a JuMP expression for A*x.\n\n\n\n\n\n","category":"method"},{"location":"#Compromise.unscale!-Tuple{Any, Compromise.AbstractAffineScaler, Any}","page":"Home","title":"Compromise.unscale!","text":"Unscale x and set ξ according to ξ = inv(T)*x - inv(T)*t.\n\n\n\n\n\n","category":"method"}]
}