# # Sharing Model Data Between Runs

# Sharing and Recycling surrogate models is not yet part of the public API.
# Below is a hack to do so nonetheless by abusing the `user_callback` functionality.

import Compromise

# The `ModelCapturer` can store a reference to the surrogate and the variable scaler:
mutable struct ModelCapturer  <: Compromise.AbstractStoppingCriterion
    mod :: Any
    scaler :: Any
    active :: Bool
end
# Its logic is to snatch that reference before the first iteration takes place …
Compromise.check_pre_iteration(crit::ModelCapturer)=crit.active
# … or copy the model parameters to the model `mod` in use, if a reference is already stored:
function Compromise.evaluate_stopping_criterion(
    crit::ModelCapturer,
    Δ, mop, mod, scaler, lin_cons, scaled_cons,
    vals, vals_tmp, step_vals, mod_vals, filter, iter_meta, step_cache, algo_opts;
)
    if isnothing(crit.scaler)
        crit.scaler = scaler
    end
    if isnothing(crit.mod)
        crit.mod = mod
        crit.active = false
    else
        Compromise.universal_copy_model!(mod, crit.mod)
        crit.active = false
    end
    return nothing
end

# Running the optimization works as usual, we just store some data in our matrices 
# `X`, `X0`, `Y`, etc. 
import HaltonSequences: HaltonPoint
import Logging
function run_problems(lb, ub, objf!; num_runs::Int, reuse_model::Bool)
    mop = Compromise.MutableMOP(; num_vars=2, lb, ub)

    Compromise.add_objectives!(
        mop, objf!, :rbf; 
        dim_out=2, func_iip=true,
    )

    algo_opts = Compromise.AlgorithmOptions(;
        log_level=Logging.Debug,
        stop_max_crit_loops=2,
    )

    user_callback = ModelCapturer(nothing, nothing, true)

    wb = ub .- lb 
    scale_x = x -> lb .+ wb .* x
    X0 = mapreduce(scale_x, hcat, HaltonPoint(2; length=num_runs))

    X = Matrix{Float64}(undef, 2, num_runs)
    Y0 = Matrix{Float64}(undef, 2, num_runs)
    Y = Matrix{Float64}(undef, 2, num_runs)
    database_x = Dict{Int, Matrix{Float64}}()
    database_y = Dict{Int, Matrix{Float64}}()

    for i = 1:num_runs
        user_callback.active=true
        if !reuse_model 
            user_callback.mod=nothing
        end
        x0 = @view(X0[:, i])
        y0 = @view(Y0[:, i])
        objf!(y0, x0)
        fv, r = Compromise.optimize(mop, x0; algo_opts, user_callback)
        X[:,i] .= fv.ξ
        Y[:,i] .= fv.fx
        db = user_callback.mod.mod_objectives.database
        db_x = db.x[:, db.database_flags_x]
        ## input sites are scaled, we have to undo that:
        database_x[i] = copy(db_x)
        for (ci,c) in enumerate(eachcol(database_x[i]))
            Compromise.unscale!(c, user_callback.scaler, db_x[:, ci])
        end
        database_y[i] = copy(db.y[:, db.database_flags_y])
    end

    return X, X0, database_x, Y, Y0, database_y
end

# ## Plotting
# Below are all the plotting functions, you can skip these definitions.
using CairoMakie
function matrix_lims(Y::Matrix)
    lims1, lims2 = extrema(Y; dims=2)
    return lims1, lims2
end

function matrix_lims(Ys...)
    lims1 = (Inf, -Inf)
    lims2 = (Inf, -Inf)
    for Y in Ys
        l1, l2 = matrix_lims(Y)
        lims1 = (min(lims1[1], l1[1]), max(lims1[2], l1[2]))
        lims2 = (min(lims2[1], l2[1]), max(lims2[2], l2[2]))
    end
    w1 = lims1[2] - lims1[1]
    w2 = lims2[2] - lims2[1]
    lims1 = (lims1[1] - w1/10, lims1[2] + w1/10)
    lims2 = (lims2[1] - w2/10, lims2[2] + w2/10)
    return lims1, lims2
end

# The animation plots a starting point in red, the final value in green, and 
# database points in orange:
function make_animation(file_name, lb, ub, X, X0, database_x, Y, Y0, database_y; ylims1=nothing, ylims2=nothing)

    pX0 = Observable(Point2[])
    pX = Observable(Point2[])
    pXdb = Observable(Point2[])
    pY0 = Observable(Point2[])
    pY = Observable(Point2[])
    pYdb = Observable(Point2[])

    fig = Figure()
    ## Plot Input Space
    ax1 = Axis(fig[1,1])
    wb = ub .- lb
    xlims!(ax1, (lb[1] - wb[1]/10 , ub[1] + wb[1]/10))
    ylims!(ax1, (lb[2] - wb[2]/10 , ub[2] + wb[2]/10))

    ## Plot Output Space
    ax2 = Axis(fig[1,2])
    if !isnothing(ylims1)
        xlims!(ax2, ylims1)
    end
    if !isnothing(ylims2)
        ylims!(ax2, ylims2)
    end

    ## make empty initial plots
    scatter!(ax1, pXdb; color=:orange, markersize=25)
    scatter!(ax1, pX0; color=:red)
    scatter!(ax1, pX; color=:green)
    
    scatter!(ax2, pYdb; color=:orange, markersize=25)
    scatter!(ax2, pY0; color=:red)
    scatter!(ax2, pY; color=:green)

    i = 1
    num_runs = size(X0, 2)
    record(fig, joinpath(@__DIR__, file_name), 1:3*num_runs; framerate = 3) do _i
        if _i % 3 == 1 
            x = Point2f(X0[1,i], X0[2,i])
            y = Point2f(Y0[1,i], Y0[2,i])
            
            pX0[] = push!(pX0[], x)
            pY0[] = push!(pY0[], y)
        elseif _i % 3 == 2
            x = Point2f(X[1,i], X[2,i])
            y = Point2f(Y[1,i], Y[2,i])
            pX[] = push!(pX[], x)
            pY[] = push!(pY[], y)
        else
            Xdb = database_x[i]
            for xj in eachcol(Xdb)
                x = Point2f(xj[1], xj[2])
                pXdb[] = push!(pXdb[], x)
            end
            Ydb = database_y[i]
            for yj in eachcol(Ydb)
                y = Point2f(yj[1], yj[2])
                pYdb[] = push!(pYdb[], y)
            end
            i+=1
        end
    end
end

# ## Defining Problems
function lovison2()
    lb = fill(-.5, 2)
    ub = [0.0, 0.5]
    objf!(y, x) = begin
        y[1] = x[2]
        y[2] = - (x[2]-x[1]^3)/(x[1]+1)
        nothing
    end
    return lb, ub, objf!
end

function paraboloids()
    lb = [-2, -2]
    ub = [2, 2]
    objf!(y, x) = begin
        y[1] = sum( (x .- 1).^2 )
        y[2] = sum( (x .+ 1).^2 )
        nothing
    end
    return lb, ub, objf!
end
# ## Lovison 2
lb, ub, objf! = lovison2()
res_noreuse = run_problems(lb, ub, objf!; num_runs=20, reuse_model=false)
res_reuse = run_problems(lb, ub, objf!; num_runs=20, reuse_model=true)

ylims1, ylims2 = matrix_lims(
    res_noreuse[4], res_noreuse[5], res_reuse[4], res_reuse[5])

make_animation("lovison2_noreuse.mp4", lb, ub, res_noreuse...; ylims1, ylims2);
# ![Lovison 2 -- No Data Sharing](lovison2_noreuse.mp4)
make_animation("lovison2_reuse.mp4", lb, ub, res_reuse...; ylims1, ylims2);
# ![Lovison 2 -- Data Sharing](lovison2_reuse.mp4)

# ## 2 Paraboloids
lb, ub, objf! =paraboloids()
res_noreuse = run_problems(lb, ub, objf!; num_runs=20, reuse_model=false)
res_reuse = run_problems(lb, ub, objf!; num_runs=20, reuse_model=true)

ylims1, ylims2 = matrix_lims(
    res_noreuse[4], res_noreuse[5], res_reuse[4], res_reuse[5])

make_animation("paraboloids_noreuse.mp4", lb, ub, res_noreuse...; ylims1, ylims2);
# ![Paraboloids -- No Data Sharing](paraboloids_noreuse.mp4)
make_animation("paraboloids_reuse.mp4", lb, ub, res_reuse...; ylims1, ylims2);
# ![Paraboloids -- No Data Sharing](paraboloids_reuse.mp4)