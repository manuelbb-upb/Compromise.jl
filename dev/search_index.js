var documenterSearchIndex = {"docs":
[{"location":"README/","page":"README","title":"README","text":"EditURL = \"../literate_src/README.jl\"","category":"page"},{"location":"README/#Compromise.jl","page":"README","title":"Compromise.jl","text":"","category":"section"},{"location":"README/","page":"README","title":"README","text":"(Image: Stable) (Image: Dev)","category":"page"},{"location":"README/#About-“CoMPrOMISE”","page":"README","title":"About “CoMPrOMISE”","text":"","category":"section"},{"location":"README/","page":"README","title":"README","text":"Constrained Multiobjective Problem Optimizer with Model Information to Save Evaluations","category":"page"},{"location":"README/","page":"README","title":"README","text":"This package provides a flexible first-order solver for constrained and unconstrained nonlinear multi-objective problems. It uses a trust region approach and either exact derivatives or local surrogate models for a derivative-free descent. Box constraints are respected during model construction and treated as unrelaxable. Box constraints and linear constraints are supported and passed down to an inner LP solver. Nonlinear constraint functions can be modelled and are dealt with by incorporating a filter. They are relaxable, i.e., all other functions must be computable even when the constraints are violated.","category":"page"},{"location":"README/#Release-Notes","page":"README","title":"Release Notes","text":"","category":"section"},{"location":"README/","page":"README","title":"README","text":"I don't really keep up a consistent versioning scheme. But the changes in this section have been significant enough to warrant some comments.","category":"page"},{"location":"README/#Version-0.2.0","page":"README","title":"Version 0.2.0","text":"","category":"section"},{"location":"README/","page":"README","title":"README","text":"We import and re-export @set and @reset from Accessors.jl.\nAlgorithmOptions is no immutable and type-stable. @set algo_opts.float_type will trigger cnoversion and setting of type-dependent defaults.\nLikewise, RBFConfig is no longer mutable and has concrete types.\nTypedMOP supports @set and @reset for objectives, nl_eq_constraints and nl_eq_constraints.\nThe AbstractNonlinearOperator interface now requires CompromiseEvaluators.operator_dim_in and CompromiseEvaluators.operator_dim_out.\nThe ReturnObject now references the whole cache (basically a NamedTuple of internal structs.)\nSimpleMOP reset call counters by default.","category":"page"},{"location":"README/#Version-0.1.0","page":"README","title":"Version 0.1.0","text":"","category":"section"},{"location":"README/","page":"README","title":"README","text":"This release is breaking, because the the RBF database is no longer thread-safe by default. Instead, ConcurrentUtils is a weak dependency and no longer mandatory. To use a thread-safe RBF database, either configure your problem functions with :rbfLocked, use an RBFConfig with database_rwlock = ConcurrentRWLock() or pre-initialize a thread-safe database by setting the field rwlock.","category":"page"},{"location":"README/#Version-0.0.3","page":"README","title":"Version 0.0.3","text":"","category":"section"},{"location":"README/","page":"README","title":"README","text":"Internally, there have been major changes regarding the caching of MOP and surrogate result values. Previously, separate preallocation functions were required (e.g., prealloc_fx …). Now, there is only init_value_caches, and instead of accessing the cache arrays as properties, there are dedicated getter methods.","category":"page"},{"location":"README/#Version-0.0.2","page":"README","title":"Version 0.0.2","text":"","category":"section"},{"location":"README/#RBF-Surrogate-Models","page":"README","title":"RBF Surrogate Models","text":"","category":"section"},{"location":"README/","page":"README","title":"README","text":"The internals of how RBF Surrogates are constructed has been redone. As before, the construction is based off “Derivative-Free Optimization Algorithms For Computationally Expensive Functions,” (Wild, 2009). In the old version, I did not really care for the matrix factorizations. Finding a poised set of points for fully-linear interpolation needs repeated QR factorizations of the point matrix. Afterwards, additional points are found by updating the Cholesky factorization of some symmetric matrix product involving the RBF Gram matrix.","category":"page"},{"location":"README/","page":"README","title":"README","text":"To save allocations, the QR factorizations now make use of structures similar to","category":"page"},{"location":"README/","page":"README","title":"README","text":"those in FastLapackInterface.   Once I manage to make a pull request   to avoid even more allocations, we can also make FastLapackInterface a dependency.","category":"page"},{"location":"README/","page":"README","title":"README","text":"The Cholesky factorization is now updated by using the formulas from the reference, and the factors are used to compute the surrogate coefficients.\nIn both cases, buffers are pre-allocated to support a maximum number of interpolation points, and we work with views mainly. Such temporary buffers are stored in RBFTrainingBuffers.\nAn RBFModel now only needs RBFParameters for successful training and reproducible evaluation. Most importantly, evaluation is decoupled from the RBFDatabase!! In older versions, we would view into the database to query interpolation points. These are now copied instead, so that changes to the database don't invalidate a model.\nWith commit cc709fa we can thus share a database in multiple optimization runs.\nAs an aside, there is a whole new backend for RBFs, which can be used in a standalone manner, too.","category":"page"},{"location":"README/","page":"README","title":"README","text":"For most of the RBF related changes, commit ab5cba8 is most relevant.","category":"page"},{"location":"README/#Other-changes","page":"README","title":"Other changes","text":"","category":"section"},{"location":"README/","page":"README","title":"README","text":"There likely was a bug in how descent directions were computed. In old versions, I tried to avoid the computation of an initial steplength by making it part of the descent direction sub-problem, but accounting for the change in criticality measure did not work out well. Commit f1386c2 makes things look a bit more elegant.\nAt some point in time, I messed up affine scaling. Should work now, and there is tests for it.\nThreaded parallel execution is now supported internally (but improvised).\nLots of tests.\nChanges to AbstractNonlinearOperator interface. A new AbstractNonlinearOperatorWrapper <: AbstractNonlinearOperator.\nNew defaults in AlgorithmOptions. Stopping based mainly on minimum radius.\nA new return type (ReturnObject).","category":"page"},{"location":"README/#Example","page":"README","title":"Example","text":"","category":"section"},{"location":"README/#Constrained-Two-Parabolas-Problem","page":"README","title":"Constrained Two-Parabolas Problem","text":"","category":"section"},{"location":"README/","page":"README","title":"README","text":"This first example uses Taylor Polynomials to approximate the function locally. For that we need a gradient function. But we also show, how to add functions with derivative-free surrogates – in this case nonlinear constraints.","category":"page"},{"location":"README/","page":"README","title":"README","text":"First we load the optimizer package, “Compromise.jl”:","category":"page"},{"location":"README/","page":"README","title":"README","text":"using Compromise","category":"page"},{"location":"README/","page":"README","title":"README","text":"The package exports a simple problem structure, MutableMOP. As the name suggests, this is a mutable structure to define a multi-objective optimization problem. We can use it to set up a problem step by step. The only information required for initialization is the number of variables:","category":"page"},{"location":"README/","page":"README","title":"README","text":"mop = MutableMOP(;num_vars = 2)","category":"page"},{"location":"README/","page":"README","title":"README","text":"For a MutableMOP, all functions are vector-to-vector. We can define the objectives (objectives), nonlinear inequality constraints (nl_ineq_constraints) and nonlinear equality constraints (nl_eq_constraints). By default, these fields default to nothing. Alternatively, they need objects of type Compromise.NonlinearFunction. We have helpers to support normal Julia functions. For example, consider this vector-to-vector function:","category":"page"},{"location":"README/","page":"README","title":"README","text":"function objective_function(x)\n    return [\n        (x[1] - 2)^2 + (x[2] - 1)^2;\n        (x[1] - 2)^2 + (x[2] + 1)^2\n    ]\nend","category":"page"},{"location":"README/","page":"README","title":"README","text":"We can easily derive the gradients, so let's also define them manually, to use derivative-based models:","category":"page"},{"location":"README/","page":"README","title":"README","text":"function objective_grads(x)\n    # return the transposed Jacobian, i.e., gradients as columns\n    df11 = df21 = 2 * (x[1] - 2)\n    df12 = 2 * (x[2] - 1)\n    df22 = 2 * (x[2] + 1)\n    return [ df11 df21; df12 df22 ]\nend","category":"page"},{"location":"README/","page":"README","title":"README","text":"To add these functions to mop, we call the helper method add_objectives and also specify the model class to be used. There are shorthand symbols, for example :exact or taylor1, for objectives with known gradients. We also have to tell the optimizer about the function signature. func_iip=true would imply an in-place objective with signature objective_function!(fx, x). dim_out is a mandatory argument.","category":"page"},{"location":"README/","page":"README","title":"README","text":"add_objectives!(\n    mop, objective_function, objective_grads, :taylor1;\n    dim_out=2, func_iip=false, grads_iip=false\n)","category":"page"},{"location":"README/","page":"README","title":"README","text":"note: Note\nFor the above objective function, it would be sensible to additionally have a function objective_values_and_grads, that returns the objectives and gradients at the same time. That is possible, MutableMOP has an interface for such optimizations.","category":"page"},{"location":"README/","page":"README","title":"README","text":"We support non-convex, nonlinear constraints (as long as they are relaxable). For example, we can constrain the problem to ℝ² without unit ball. For demonstration purposes, use an in-place function:","category":"page"},{"location":"README/","page":"README","title":"README","text":"nl_ineq_function!(y, x) = y[1] = 1 - sum(x.^2)","category":"page"},{"location":"README/","page":"README","title":"README","text":"Of course, that is a fairly simple constraint function. If it was more complicated, we could be tempted to use automatic differentiation for derivative calculations. Instead, you can also use derivative-free models, such as radial basis function (RBF) models.","category":"page"},{"location":"README/","page":"README","title":"README","text":"For now, we stick with the fixed shape parameter and finalize our problem:","category":"page"},{"location":"README/","page":"README","title":"README","text":"add_nl_ineq_constraints!(mop, nl_ineq_function!, :rbf;\n    func_iip=true, dim_out=1\n)","category":"page"},{"location":"README/","page":"README","title":"README","text":"The MutableMOP is turned into a TypedMOP during initialization. We can thus simply pass it to optimize:","category":"page"},{"location":"README/","page":"README","title":"README","text":"ret = optimize(mop, [-2.0, 0.5])","category":"page"},{"location":"README/","page":"README","title":"README","text":"ret is the return object. You can query it using functions like opt_vars etc. Final argument vector:","category":"page"},{"location":"README/","page":"README","title":"README","text":"opt_vars(ret)","category":"page"},{"location":"README/","page":"README","title":"README","text":"Final value vector:","category":"page"},{"location":"README/","page":"README","title":"README","text":"opt_objectives(ret)","category":"page"},{"location":"README/","page":"README","title":"README","text":"Final constraint vector:","category":"page"},{"location":"README/","page":"README","title":"README","text":"opt_nl_ineq_constraints(ret)","category":"page"},{"location":"README/#More-RBF-Options","page":"README","title":"More RBF Options","text":"","category":"section"},{"location":"README/","page":"README","title":"README","text":"Instead of passing :rbf, you can also pass an RBFConfig. To use the Gaussian kernel:","category":"page"},{"location":"README/","page":"README","title":"README","text":"cfg = RBFConfig(; kernel=GaussianKernel())","category":"page"},{"location":"README/","page":"README","title":"README","text":"Or the inverse multiquadric:","category":"page"},{"location":"README/","page":"README","title":"README","text":"cfg = RBFConfig(; kernel=InverseMultiQuadricKernel())","category":"page"},{"location":"README/","page":"README","title":"README","text":"Then:","category":"page"},{"location":"README/","page":"README","title":"README","text":"mop = MutableMOP(; num_vars=2)\nadd_objectives!(\n    mop, objective_function, cfg; dim_out=2, func_iip=false,\n)\nret = optimize(mop, [-2.0, 0.5])","category":"page"},{"location":"README/","page":"README","title":"README","text":"See the docstring for more options.","category":"page"},{"location":"README/#Sharing-an-RBFDatabase","page":"README","title":"Sharing an RBFDatabase","text":"","category":"section"},{"location":"README/","page":"README","title":"README","text":"Normally, each optimization run initializes a new database. But a database is only ever referenced. We can thus pre-initialize a database and use it in multiple runs:","category":"page"},{"location":"README/","page":"README","title":"README","text":"rbf_db = Compromise.RBFModels.init_rbf_database(2, 2, nothing, nothing, Float64)\ncfg = RBFConfig(; database=rbf_db)","category":"page"},{"location":"README/","page":"README","title":"README","text":"Set up problem:","category":"page"},{"location":"README/","page":"README","title":"README","text":"mop = MutableMOP(; num_vars=2)\nobjf_counter = Ref(0)\nfunction counted_objf(x)\n    global objf_counter[] += 1\n    return objective_function(x)\nend\nadd_objectives!(\n    mop, counted_objf, cfg; dim_out=2, func_iip=false,\n)","category":"page"},{"location":"README/","page":"README","title":"README","text":"First run:","category":"page"},{"location":"README/","page":"README","title":"README","text":"ret = optimize(mop, [-2.0, 0.5])\nobjf_counter[]","category":"page"},{"location":"README/","page":"README","title":"README","text":"Second run:","category":"page"},{"location":"README/","page":"README","title":"README","text":"ret = optimize(mop, [-2.0, 0.0])\nobjf_counter[]","category":"page"},{"location":"README/#Parallelism","page":"README","title":"Parallelism","text":"","category":"section"},{"location":"README/","page":"README","title":"README","text":"The RBF update algorithm has a lock to access the database in a safe way (?) when multiple optimization runs are done concurrently. There even is an “algorithm” for this:","category":"page"},{"location":"README/","page":"README","title":"README","text":"using ConcurrentUtils\nmop = MutableMOP(; num_vars=2)\nadd_objectives!(\n    mop, counted_objf, :rbfLocked; dim_out=2, func_iip=false,\n)\nX0 = [\n    -2.0    -2.0    0.0\n    0.5     0.0     0.0\n]\nopts = Compromise.ThreadedOuterAlgorithmOptions(;\n    inner_opts=AlgorithmOptions(;\n        stop_delta_min = 1e-8,\n    )\n)\nrets = Compromise.optimize_with_algo(mop, opts, X0)","category":"page"},{"location":"README/#Stopping-based-on-Number-of-Function-Evaluations","page":"README","title":"Stopping based on Number of Function Evaluations","text":"","category":"section"},{"location":"README/","page":"README","title":"README","text":"The restriction of evaluation budget is a property of the evaluators. Because of this, it is not configurable with AlgorithmOptions. You can pass max_func_calls as a keyword argument to add_objectives! and similar functions. Likewise, max_grad_calls restricts the number of gradient calls, max_hess_calls limits Hessian computations.","category":"page"},{"location":"README/","page":"README","title":"README","text":"~~For historic reasons, the count is kept between runs.~~ The count is now reset between runs by default. To reset the count between runs (sequential or parallel), indicate it when setting up the MOP.","category":"page"},{"location":"README/","page":"README","title":"README","text":"mop = MutableMOP(; num_vars=2, reset_call_counters=false)   # default\nadd_objectives!(\n    mop, objective_function, :rbf; dim_out=2, func_iip=false, max_func_calls=10\n)\nret1 = optimize(mop, [-2, .5])","category":"page"},{"location":"README/","page":"README","title":"README","text":"Now, there is no budget left for a second run:","category":"page"},{"location":"README/","page":"README","title":"README","text":"ret2 = optimize(mop, [-2, -.5])\nismissing(opt_vars(ret2))","category":"page"},{"location":"README/","page":"README","title":"README","text":"Here is a remedy:","category":"page"},{"location":"README/","page":"README","title":"README","text":"mop.reset_call_counters=true\nret1 = optimize(mop, [-2, .5])","category":"page"},{"location":"README/","page":"README","title":"README","text":"Now, there is budget left for a second run:","category":"page"},{"location":"README/","page":"README","title":"README","text":"ret2 = optimize(mop, [-2, -.5])\n!ismissing(opt_vars(ret2))","category":"page"},{"location":"README/#Automatic-Differentiation","page":"README","title":"Automatic Differentiation","text":"","category":"section"},{"location":"README/","page":"README","title":"README","text":"There is an optional ForwardDiff extension. To use a derivative-based model without specifying the gradients by-hand, first load ForwardDiff.","category":"page"},{"location":"README/","page":"README","title":"README","text":"using ForwardDiff","category":"page"},{"location":"README/","page":"README","title":"README","text":"Now, ForwardDiffBackend should be available:","category":"page"},{"location":"README/","page":"README","title":"README","text":"diff_backend = ForwardDiffBackend()","category":"page"},{"location":"README/","page":"README","title":"README","text":"Set up the problem:","category":"page"},{"location":"README/","page":"README","title":"README","text":"mop = MutableMOP(2)\nadd_objectives!(mop, objective_function, :exact;\n    func_iip=false, dim_out=2, backend=diff_backend\n)\n\noptimize(mop, -5 .+ 10 .* rand(2))","category":"page"},{"location":"README/","page":"README","title":"README","text":"","category":"page"},{"location":"README/","page":"README","title":"README","text":"This page was generated using Literate.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = Compromise","category":"page"},{"location":"#Compromise","page":"Home","title":"Compromise","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for Compromise. There is not much here yet. Everything is still very much a work-in-progress.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Random Doc-Strings:","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [Compromise.NonlinearFunctions, Compromise,]","category":"page"},{"location":"#Compromise.NonlinearFunctions.NonlinearParametricFunction","page":"Home","title":"Compromise.NonlinearFunctions.NonlinearParametricFunction","text":"NonlinearParametricFunction(; \n    func, grads=nothing, hessians=nothing, \n    func_and_grads=nothing, func_and_grads_and_hessians=nothing,\n    backend=nothing,\n    func_iip=true, grads_iip=true, hessians_iip=true, \n    func_and_grads_iip=true, func_and_grads_and_hessians_iip=true)\n\nA flexible function wrapper to conveniently query evaluations and derivatives of user provided functions.\n\nIf the user provided function is used in a derivative-free alogrithm, only func has  to be provided. The flag func_iip indicates its signature: If func_iip==true, the function should mutate the target array and  have signature func!(y, x, p). Otherwise, y = func(x, p).\n\nShould gradients be needed, function handles can be provided, and the respective  flags indicate the following signatures:\n\ngrads_iip == true implies grads!(Dy, x, p), otherwise Dy = grads(x, p).\nhessians_iip == true implies hessians!(H, x, p), otherwise H = hessians(x, p).\nfunc_and_grads_iip == true implies func_and_grads!(y, Dy, x, p),    otherwise y, Dy = func_and_grads(x, p).\nfunc_and_grads_and_hessians_iip == true implies   func_and_grads_and_hessians!(y, Dy, H, x, p),    else y, Dy, H = func_and_grads_and_hessians(x, p).\n\nAlternatively (or additionally), an AbstractAutoDiffBackend can be passed to  compute the derivatives if the relevant field isnothing.\n\n\n\n\n\n","category":"type"},{"location":"#Compromise.AlgorithmOptions","page":"Home","title":"Compromise.AlgorithmOptions","text":"AlgorithmOptions(; kwargs...)\n\nConfigure the optimization by passing keyword arguments:\n\nfloat_type::Type{T} where T<:AbstractFloat\nstep_config::Any: Configuration object for descent and normal step computation.\nscaler_cfg::Any: Configuration to determine variable scaling (if model supports it). Either :box or :none.\nrequire_fully_linear_models::Bool\nlog_level::Base.CoreLogging.LogLevel: Control verbosity by setting a min. level for @logmsg.\nmax_iter::Int64: Maximum number of iterations.\nstop_delta_min::Compromise.NumberWithDefault{T} where T<:AbstractFloat: Stop if the trust region radius is reduced to below stop_delta_min.\nstop_xtol_rel::AbstractFloat: Stop if the trial point xₜ is accepted and xₜ - x δx.\nstop_xtol_abs::AbstractFloat: Stop if the trial point xₜ is accepted and xₜ - x ε.\nstop_ftol_rel::AbstractFloat: Stop if the trial point xₜ is accepted and f(xₜ) - f(x) δf(x).\nstop_ftol_abs::AbstractFloat: Stop if the trial point xₜ is accepted and f(xₜ) - f(x) ε.\nstop_crit_tol_abs::AbstractFloat: Stop if for the approximate criticality it holds that χ(x) = ε and for the feasibility that θ = δ.\nstop_theta_tol_abs::Compromise.NumberWithDefault{T} where T<:AbstractFloat: Stop if for the approximate criticality it holds that χ(x) = ε and for the feasibility that θ = δ.\nstop_max_crit_loops::Int64: Stop after the criticality routine has looped stop_max_crit_loops times.\neps_crit::AbstractFloat: Lower bound for criticality before entering Criticality Routine.\neps_theta::AbstractFloat: Lower bound for feasibility before entering Criticality Routine.\ncrit_B::AbstractFloat: At the end of the Criticality Routine the radius is possibly set to crit_B * χ.\ncrit_M::AbstractFloat: Criticality Routine runs until Δ ≤ crit_M * χ.\ncrit_alpha::AbstractFloat: Trust region shrinking factor in criticality loops.\nbacktrack_in_crit_routine::Bool\ndelta_init::AbstractFloat: Initial trust region radius.\ndelta_max::AbstractFloat: Maximum trust region radius.\ngamma_shrink_much::AbstractFloat: Most severe trust region reduction factor.\ngamma_shrink::AbstractFloat: Trust region reduction factor.\ngamma_grow::AbstractFloat: Trust region enlargement factor.\nstrict_acceptance_test::Bool: Whether to require all objectives to be reduced or not.\nnu_accept::AbstractFloat: Acceptance threshold.\nnu_success::AbstractFloat: Success threshold.\nc_delta::AbstractFloat: Factor for normal step compatibility test. The smaller c_delta, the stricter the test.\nc_mu::AbstractFloat: Factor for normal step compatibility test. The smaller c_mu, the stricter the test for small radii.\nmu::AbstractFloat: Exponent for normal step compatibility test. The larger mu, the stricter the test for small radii.\nkappa_theta::AbstractFloat: Factor in the model decrease condition.\npsi_theta::AbstractFloat: Exponent (for constraint violation) in the model decrease condition.\nnl_opt::Symbol: NLopt algorithm symbol for restoration phase.\n\n\n\n\n\n","category":"type"},{"location":"#Compromise.MutableMOP","page":"Home","title":"Compromise.MutableMOP","text":"MutableMOP(; num_vars, kwargs...)\n\nInitialize a multi-objective problem with num_vars variables.\n\nFunctions\n\nThere can be exactly one (possibly vector-valued) objective function, one nonlinear equality constraint function, and one nonlinear inequality constraint function. For now, they have to be of type NonlinearFunction. You could provide these functions with the keyword-arguments objectives, nl_eq_constraints or nl_ineq_constraints or set the fields of the same name. To conveniently add user-provided functions, there are helper functions,  like add_objectives!.\n\nLinearConstraints\n\nBox constraints are defined by the vectors lb and ub. Linear equality constraints Ex=c are defined by the matrix E and the vector c. Inequality constraints read Axb and use A and b.\n\nSurrogate Configuration\n\nUse the keyword-arguments mcfg_objectives to provide an AbstractSurrogateModelConfig to define how the objectives should be modelled. By default, we assume ExactModelConfig(), which requires differentiable objectives.\n\n\n\n\n\n","category":"type"},{"location":"#Compromise.SimpleValueCache","page":"Home","title":"Compromise.SimpleValueCache","text":"A struct holding values computed for or derived from an AbstractMOP.\n\n\n\n\n\n","category":"type"},{"location":"#Compromise.add_function!-Union{Tuple{field}, Tuple{Val{field}, MutableMOP, Compromise.CompromiseEvaluators.AbstractNonlinearOperator}, Tuple{Val{field}, MutableMOP, Compromise.CompromiseEvaluators.AbstractNonlinearOperator, Union{Nothing, Compromise.CompromiseEvaluators.AbstractSurrogateModelConfig, Symbol}}} where field","page":"Home","title":"Compromise.add_function!","text":"add_function!(func_field, mop, op, model_cfg; dim_out, backend=NoBackend())\n\nAdd the operator op to mop at func_field and use model configuration model_cfg. Keyword argument dim_out::Int is mandatory. E.g., add_function!(:objectives, mop, op, :rbf; dim_out=2) adds op as the bi-valued objective to mop.\n\n\n\n\n\n","category":"method"},{"location":"#Compromise.add_nl_eq_constraints!","page":"Home","title":"Compromise.add_nl_eq_constraints!","text":"add_nl_eq_constraints!(mop::MutableMOP, func, grads, model_cfg=nothing; \n    dim_out::Int, kwargs...)\n\nSet function func to return the nonlinear equality constraints vector of mop. Argument model_cfg is optional and specifies the surrogate models for func. Can be nothing, a Symbol (:exact, :rbf, taylor1, taylor2), or an AbstractSurrogateModelConfig object. grads should be a function mapping a vector to the transposed jacobian of func.\n\nAll functions can be in-place, see keyword arguments func_iip and grads_iip.\n\nKeyword argument dim_out is mandatory and corresponds to the length of the result vector. The other kwargs... are passed to the inner AbstractNonlinearOperator as is. For options and defaults see NonlinearParametricFunction.\n\n\n\n\n\n","category":"function"},{"location":"#Compromise.add_nl_eq_constraints!-2","page":"Home","title":"Compromise.add_nl_eq_constraints!","text":"add_nl_eq_constraints!(mop::MutableMOP, func, model_cfg=nothing; \n    dim_out::Int, kwargs...)\n\nSet function func to return the nonlinear equality constraints vector of mop. Argument model_cfg is optional and specifies the surrogate models for func. Can be nothing, a Symbol (:exact, :rbf, taylor1, taylor2), or an AbstractSurrogateModelConfig object.\n\nAll functions can be in-place, see keyword argument func_iip.\n\nKeyword argument dim_out is mandatory and corresponds to the length of the result vector. If dim_vars(mop) <= 0, then dim_in is also mandatory. The other kwargs... are passed to the inner AbstractNonlinearOperator as is. For options and defaults see NonlinearParametricFunction.\n\n\n\n\n\n","category":"function"},{"location":"#Compromise.add_nl_eq_constraints!-3","page":"Home","title":"Compromise.add_nl_eq_constraints!","text":"add_nl_eq_constraints!(mop::MutableMOP, func, grads, func_and_grads, model_cfg=nothing; \n    dim_out::Int, kwargs...)\n\nSet function func to return the nonlinear equality constraints vector of mop. Argument model_cfg is optional and specifies the surrogate models for func. Can be nothing, a Symbol (:exact, :rbf, taylor1, taylor2), or an AbstractSurrogateModelConfig object. grads should be a function mapping a vector to the transposed jacobian of func, while func_and_grads returns a primal vector and the gradients at the same time.\n\nAll functions can be in-place, see keyword arguments func_iip, grads_iip and  func_and_grads_iip.\n\nKeyword argument dim_out is mandatory and corresponds to the length of the result vector. The other kwargs... are passed to the inner AbstractNonlinearOperator as is. For options and defaults see NonlinearParametricFunction.\n\n\n\n\n\n","category":"function"},{"location":"#Compromise.add_nl_ineq_constraints!","page":"Home","title":"Compromise.add_nl_ineq_constraints!","text":"add_nl_ineq_constraints!(mop::MutableMOP, func, grads, model_cfg=nothing; \n    dim_out::Int, kwargs...)\n\nSet function func to return the nonlinear inequality constraints vector of mop. Argument model_cfg is optional and specifies the surrogate models for func. Can be nothing, a Symbol (:exact, :rbf, taylor1, taylor2), or an AbstractSurrogateModelConfig object. grads should be a function mapping a vector to the transposed jacobian of func.\n\nAll functions can be in-place, see keyword arguments func_iip and grads_iip.\n\nKeyword argument dim_out is mandatory and corresponds to the length of the result vector. The other kwargs... are passed to the inner AbstractNonlinearOperator as is. For options and defaults see NonlinearParametricFunction.\n\n\n\n\n\n","category":"function"},{"location":"#Compromise.add_nl_ineq_constraints!-2","page":"Home","title":"Compromise.add_nl_ineq_constraints!","text":"add_nl_ineq_constraints!(mop::MutableMOP, func, model_cfg=nothing; \n    dim_out::Int, kwargs...)\n\nSet function func to return the nonlinear inequality constraints vector of mop. Argument model_cfg is optional and specifies the surrogate models for func. Can be nothing, a Symbol (:exact, :rbf, taylor1, taylor2), or an AbstractSurrogateModelConfig object.\n\nAll functions can be in-place, see keyword argument func_iip.\n\nKeyword argument dim_out is mandatory and corresponds to the length of the result vector. If dim_vars(mop) <= 0, then dim_in is also mandatory. The other kwargs... are passed to the inner AbstractNonlinearOperator as is. For options and defaults see NonlinearParametricFunction.\n\n\n\n\n\n","category":"function"},{"location":"#Compromise.add_nl_ineq_constraints!-3","page":"Home","title":"Compromise.add_nl_ineq_constraints!","text":"add_nl_ineq_constraints!(mop::MutableMOP, func, grads, func_and_grads, model_cfg=nothing; \n    dim_out::Int, kwargs...)\n\nSet function func to return the nonlinear inequality constraints vector of mop. Argument model_cfg is optional and specifies the surrogate models for func. Can be nothing, a Symbol (:exact, :rbf, taylor1, taylor2), or an AbstractSurrogateModelConfig object. grads should be a function mapping a vector to the transposed jacobian of func, while func_and_grads returns a primal vector and the gradients at the same time.\n\nAll functions can be in-place, see keyword arguments func_iip, grads_iip and  func_and_grads_iip.\n\nKeyword argument dim_out is mandatory and corresponds to the length of the result vector. The other kwargs... are passed to the inner AbstractNonlinearOperator as is. For options and defaults see NonlinearParametricFunction.\n\n\n\n\n\n","category":"function"},{"location":"#Compromise.add_objectives!","page":"Home","title":"Compromise.add_objectives!","text":"add_objectives!(mop::MutableMOP, func, grads, model_cfg=nothing; \n    dim_out::Int, kwargs...)\n\nSet function func to return the objectives vector of mop. Argument model_cfg is optional and specifies the surrogate models for func. Can be nothing, a Symbol (:exact, :rbf, taylor1, taylor2), or an AbstractSurrogateModelConfig object. grads should be a function mapping a vector to the transposed jacobian of func.\n\nAll functions can be in-place, see keyword arguments func_iip and grads_iip.\n\nKeyword argument dim_out is mandatory and corresponds to the length of the result vector. The other kwargs... are passed to the inner AbstractNonlinearOperator as is. For options and defaults see NonlinearParametricFunction.\n\n\n\n\n\n","category":"function"},{"location":"#Compromise.add_objectives!-2","page":"Home","title":"Compromise.add_objectives!","text":"add_objectives!(mop::MutableMOP, func, model_cfg=nothing; \n    dim_out::Int, kwargs...)\n\nSet function func to return the objectives vector of mop. Argument model_cfg is optional and specifies the surrogate models for func. Can be nothing, a Symbol (:exact, :rbf, taylor1, taylor2), or an AbstractSurrogateModelConfig object.\n\nAll functions can be in-place, see keyword argument func_iip.\n\nKeyword argument dim_out is mandatory and corresponds to the length of the result vector. If dim_vars(mop) <= 0, then dim_in is also mandatory. The other kwargs... are passed to the inner AbstractNonlinearOperator as is. For options and defaults see NonlinearParametricFunction.\n\n\n\n\n\n","category":"function"},{"location":"#Compromise.add_objectives!-3","page":"Home","title":"Compromise.add_objectives!","text":"add_objectives!(mop::MutableMOP, func, grads, func_and_grads, model_cfg=nothing; \n    dim_out::Int, kwargs...)\n\nSet function func to return the objectives vector of mop. Argument model_cfg is optional and specifies the surrogate models for func. Can be nothing, a Symbol (:exact, :rbf, taylor1, taylor2), or an AbstractSurrogateModelConfig object. grads should be a function mapping a vector to the transposed jacobian of func, while func_and_grads returns a primal vector and the gradients at the same time.\n\nAll functions can be in-place, see keyword arguments func_iip, grads_iip and  func_and_grads_iip.\n\nKeyword argument dim_out is mandatory and corresponds to the length of the result vector. The other kwargs... are passed to the inner AbstractNonlinearOperator as is. For options and defaults see NonlinearParametricFunction.\n\n\n\n\n\n","category":"function"},{"location":"#Compromise.diff_mod!-Tuple{Compromise.AbstractMOPSurrogateCache, Compromise.AbstractMOPSurrogate, Any}","page":"Home","title":"Compromise.diff_mod!","text":"Evaluate the model gradients of mod at x and store results in mod_vals::SurrogateValueArrays.\n\n\n\n\n\n","category":"method"},{"location":"#Compromise.eval_and_diff_mod!-Tuple{Compromise.AbstractMOPSurrogateCache, Any, Any}","page":"Home","title":"Compromise.eval_and_diff_mod!","text":"Evaluate and differentiate mod at x and store results in mod_vals::SurrogateValueArrays.\n\n\n\n\n\n","category":"method"},{"location":"#Compromise.eval_mod!-Tuple{Compromise.AbstractMOPSurrogateCache, Compromise.AbstractMOPSurrogate, Any}","page":"Home","title":"Compromise.eval_mod!","text":"eval_mod!(mod_cache::AbstractMOPSurrogateCache, mod::AbstractMOPSurrogate, x)\n\nEvaluate mod at x and update cache mod_cache.\n\n\n\n\n\n","category":"method"},{"location":"#Compromise.eval_mop!-NTuple{9, Any}","page":"Home","title":"Compromise.eval_mop!","text":"Evaluate mop at unscaled site ξ and modify result arrays in place.\n\n\n\n\n\n","category":"method"},{"location":"#Compromise.intersect_bound-Union{Tuple{B}, Tuple{Z}, Tuple{X}, Tuple{X, Z, B}, Tuple{X, Z, B, Any}} where {X<:Number, Z<:Number, B<:Number}","page":"Home","title":"Compromise.intersect_bound","text":"intersect_bound(xi, zi, bi)\n\nGiven number xi, zi and bi, compute and return an interval  I (a tuple with 2 elements) such that xi + σ * zi <= bi is true for all σ in I.  If the constraint is feasible, at least one of the interval elements is infinite. If it is infeasible, (NaN, NaN) is returned.\n\n\n\n\n\n","category":"method"},{"location":"#Compromise.intersect_box-Union{Tuple{U}, Tuple{L}, Tuple{Z}, Tuple{X}, Tuple{X, Z, L, U}} where {X, Z, L, U}","page":"Home","title":"Compromise.intersect_box","text":"intersect_box(x, z, lb, ub)\n\nGiven vectors x, z, lb and ub, compute and return the largest  interval I (a tuple with 2 elements) such that  lb .<= x .+ σ .* z .<= ub is true for all σ in I.  If the constraints are not feasible, (NaN, NaN) is returned. If the direction z is zero, the interval could contain infinite elements.\n\n\n\n\n\n","category":"method"},{"location":"#Compromise.intersect_intervals","page":"Home","title":"Compromise.intersect_intervals","text":"Helper to intersect to intervals.\n\n\n\n\n\n","category":"function"},{"location":"#Compromise.scale_eq!-Tuple{AbstractMatrix, AbstractVector, Any}","page":"Home","title":"Compromise.scale_eq!","text":"Make Aξ ? b applicable in scaled domain.\n\n\n\n\n\n","category":"method"},{"location":"#Compromise.trust_region_bounds!-NTuple{4, Any}","page":"Home","title":"Compromise.trust_region_bounds!","text":"trust_region_bounds!(lb, ub, x, Δ)\n\nMake lb the lower left corner of a trust region hypercube with  radius Δ and make ub the upper right corner.\n\n\n\n\n\n","category":"method"},{"location":"#Compromise.trust_region_bounds!-NTuple{6, Any}","page":"Home","title":"Compromise.trust_region_bounds!","text":"trust_region_bounds!(lb, ub, x, Δ, global_lb, global_ub)\n\nMake lb the lower left corner of a trust region hypercube with  radius Δ and make ub the upper right corner. global_lb and global_ub are the global bound vectors or nothing.\n\n\n\n\n\n","category":"method"},{"location":"#Compromise.var_bounds_valid-Tuple{Any, Any}","page":"Home","title":"Compromise.var_bounds_valid","text":"`var_bounds_valid(lb, ub)`\n\nReturn true if lower bounds lb and upper bounds ub are consistent.\n\n\n\n\n\n","category":"method"},{"location":"#Compromise.@forward-Tuple{Any, Any}","page":"Home","title":"Compromise.@forward","text":"@forward WrapperType.wrapped_obj fnname(arg1, fwarg::WrapperType, args...; kwargs...)\n\nDefines a new method for fnname forwarding to method dispatching on wrapped_obj.\n\n\n\n\n\n","category":"macro"},{"location":"#Compromise.@ignoraise","page":"Home","title":"Compromise.@ignoraise","text":"@ignoraise a, b, c = critical_function(args...) [indent=0]\n\nEvaluate the right-hand side. If it returns an AbstractStoppingCriterion, make sure it is wrapped and return it. Otherwise, unpack the returned values into the left-hand side.\n\nAlso valid:\n\n@ignoraise critical_function(args...)\n@ignoraise critical_function(args...) indent_var\n\nThe indent expression must evaluate to an Int.\n\n\n\n\n\n","category":"macro"},{"location":"#Compromise.@ignorebreak","page":"Home","title":"Compromise.@ignorebreak","text":"@ignorebreak ret_var = critical_function(args...)\n\nSimilar to @ignoraise, but instead of returning if critical_function returns  an AbstractStoppingCriterion, we break. This allows for post-processing before eventually returning. ret_var is optional, but in constrast to @ignoraise, we unpack unconditionally, so length of return values should match length of left-hand side expression.\n\n\n\n\n\n","category":"macro"}]
}
