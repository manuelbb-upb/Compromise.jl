import Compromise as C
import .C.RBFModels as R
import ForwardDiff as FD

import Compromise.RBFModels: add_to_database!, val, val!
import Compromise: NonlinearFunction, func_vals!, func_grads! 

using Test

import LinearAlgebra as LA

import Random

#import FiniteDiff: finite_difference_jacobian
#%%
function eval_rbf(x, φ, centers, coeff_φ, coeff_π, ε)
    dim_φ, dim_out = size(coeff_φ)
    dim_π, _dim_out = size(coeff_π)
    @assert dim_out == _dim_out

    y_π = if dim_π == 0
        0
    elseif dim_π == 1
        coeff_π
    else
        [x; 1]' * coeff_π
    end

    Δx = centers .- x
    r = LA.norm.(eachcol(Δx))
    φ = R.apply_kernel.(φ, r, ε)
    y_φ = φ' * coeff_φ

    return vec(y_π .+ y_φ)
end

function diff_rbf(x, φ, centers, coeff_φ, coeff_π, ε)
    return FD.jacobian(
        _x -> eval_rbf(_x, φ, centers, coeff_φ, coeff_π, ε), 
        x
    )   
end

function some_rbf(;
    kernel = R.GaussianKernel(),
    dim_x = 2, 
    poly_deg = 1,
    kwargs...
)
    return R.RBFSurrogate(; kernel, dim_x, poly_deg, kwargs...)
end
#%%
@testset "Least Squares Fitting" begin
    Random.seed!(314)
    
    # columns of `X` are the input datasites („features”)
    X = [
        0 1 0
        0 0 1
    ]
    # columns of `Y` are the output datasites („targets”)
    Y = rand(4, 3)

    # initialize an RBF with params
    rbf = some_rbf(;dim_φ=3, dim_y=4)
    params = R._rbf_params(rbf, T=Float64; centers=X) 
    R._rbf_fit!(params, rbf, X, Y)

    # evaluate at single sample
    y = R._rbf_eval(rbf, rand(2), params)
    @assert size(y) == (4, 1)
    # evaluate at 3 samples
    y = R._rbf_eval(rbf, rand(2, 3), params)
    @assert size(y) == (4, 3)
    
    # test interpolation
    Y_pred = R._rbf_eval(rbf, X, params)
    @test Y ≈ Y_pred 

    # test differentiation
    Dy = diff_rbf(
        zeros(2), rbf.kernel, params.centers, params.coeff_φ, params.coeff_π, params.ε)
    Dy_pred = R._rbf_diff(rbf, zeros(2), params)
    @test Dy ≈ Dy_pred'
end

#%%
@testset "Values and Diffs of RBFSurrogate" begin
    Random.seed!(2718)
    for kernel in (R.GaussianKernel(), R.InverseMultiQuadricKernel(), R.CubicKernel())
        r = rand()
        d1 = FD.derivative( _r -> R.apply_kernel(kernel, _r, 1), r)
        d2 = R.apply_kernel_derivative(kernel, r, 1)
        @test d1 ≈ d2

        for dim_x in (1, 2, 4, 10)
        for poly_deg in (nothing, 0, 1)
        for dim_y in (1, 2, 4, 10)
        for dim_φ in (1, 5, 10, 20)
            
            @debug """
            kernel = $(kernel)
            dim_x = $(dim_x)
            poly_deg = $(poly_deg)
            dim_y = $(dim_y)
            dim_φ = $(dim_φ)"""
            
            rbf = R.RBFSurrogate(; 
                kernel, dim_φ, dim_y, dim_x, poly_deg
            )
            params = R._rbf_params(rbf, T=Float64)

            centers=params.centers
            coeff_φ=params.coeff_φ
            coeff_π=params.coeff_π 
            ε=params.ε
            @test size(centers) == (dim_x, dim_φ)
            @test size(coeff_φ) == (dim_φ, dim_y)
            @test size(coeff_π) == (rbf.dim_π, dim_y)
            @test ε > 0

            x = rand(dim_x)
            y1 = eval_rbf(
                x, kernel, centers, coeff_φ, coeff_π, ε)

            y2 = R._rbf_eval(rbf, x, params)

            @test y1 ≈ y2

            Dy1 = diff_rbf(
                x, kernel, centers, coeff_φ, coeff_π, ε)

            Dy2 = R._rbf_diff(rbf, x, params)
            @test Dy1 ≈ Dy2'
        end
        end
        end
        end
    end
end

@testset "RBFConfig" begin
    cfg = R.RBFConfig()
    @test cfg.kernel isa R.CubicKernel
    @test cfg.poly_deg == 1
    @test cfg.shape_parameter_function === nothing
    @test cfg.max_points === nothing
    @test cfg.database_size === nothing
    @test cfg.database_chunk_size === nothing
    @test cfg.enforce_fully_linear
    @test cfg.search_factor == 2
    @test cfg.max_search_factor == 2
    @test cfg.th_qr == 1/4
    @test cfg.th_cholesky == 1e-7

    dim_x = 2
    dim_y = 3
    rbf = C.init_surrogate(
        cfg, nothing, dim_x, dim_y, nothing, Float64;
        delta_max = Inf,
    )
    @test rbf.dim_x == dim_x
    @test rbf.dim_y == dim_y
    @test rbf.min_points == dim_x + 1
    @test rbf.max_points == 2*(dim_x + 1)
    @test rbf.poly_deg == 1
    @test rbf.dim_π == dim_x + 1

    dim_π = rbf.dim_π
    p = rbf.params
    b = rbf.buffers
    for fn in (:dim_x, :dim_y, :dim_π, :min_points, :max_points)
        @test getfield(rbf, fn) == getfield(p, fn)
        @test getfield(rbf, fn) == getfield(b, fn)
    end
    
    @test length(p.x0) == dim_x
    @test p.n_X_ref isa C.RBFModels.MutableNumber
    @test size(p.X, 1) >= dim_x
    @test size(p.X, 2) >= rbf.min_points
    @test size(p.coeff_φ, 1) >= rbf.max_points
    @test size(p.coeff_φ, 2) >= dim_y
    @test size(p.coeff_π, 1) >= dim_π
    @test size(p.coeff_π, 2) >= dim_y

    ## TODO check buffers
end

@testset "RBFDatabase" begin
    @test fieldnames(R.RBFDatabase) == (
        :dim_x, :dim_y, :max_size, :chunk_size, :x, :y, 
        :flags_x, :flags_y, :current_size, :state, :rwlock
    )

    cfg = R.RBFConfig()
    dat = R.init_rbf_database(cfg, 2, 3, Float32)
    @test dat.dim_x == 2
    @test dat.dim_y == 3
    @test dat.max_size == 125000000
    @test dat.chunk_size == 6
    @test size(dat.x) == (2, 6)
    @test size(dat.y) == (3, 6)
    @test eltype(dat.x) == Float32
    @test all(dat.flags_x .== 0)
    @test all(dat.flags_y .== 0)
    @test length(dat.flags_x) == 6
    @test length(dat.flags_y) == 6

    cfg = R.RBFConfig(; database_size=10)
    dat = R.init_rbf_database(cfg, 2, 3, Float32)
    @test dat.max_size == 10
    @test dat.chunk_size == 6
    cfg = R.RBFConfig(; database_size=2)
    dat = R.init_rbf_database(cfg, 2, 3, Float32)
    @test dat.max_size == 2 # this used to be reset to 3, but I have commented the warning
    @test dat.chunk_size == 2

    cfg = R.RBFConfig(; database_size=4, database_chunk_size=3)
    dat = R.init_rbf_database(cfg, 2, 3, Float32)
    @test dat.max_size == 4
    @test dat.chunk_size == 3
    
    cfg = R.RBFConfig(; database_size=3, database_chunk_size=4)
    dat = R.init_rbf_database(cfg, 2, 3, Float32)
    @test dat.max_size == 3
    @test dat.chunk_size == 3
    
    cfg = R.RBFConfig(; database_size = 9, database_chunk_size=3)
    dat = R.init_rbf_database(cfg, 2, 3, Float32)
    @test size(dat.x, 2) == 3
    @test R.db_grow!(dat) == true
    @test size(dat.x, 2) == 6
    @test size(dat.y, 2) == 6
    @test length(dat.flags_x) == 6
    @test length(dat.flags_y) == 6
    @test R.db_grow!(dat) == true
    @test size(dat.x, 2) == 9
    @test R.db_grow!(dat) == false
    @test R.db_grow!(dat; force_chunk = 2) == true
    @test size(dat.x, 2) == 11

    for i=1:11
        x = fill(i, 2)
        ix = R.add_to_database!(dat, x)
        @test i == ix
    end
    ix = R.add_to_database!(dat, rand(2))
    @test ix == 1

    ix = R.add_to_database!(dat, rand(2), rand(3))
    @test ix == 2
    ix = R.add_to_database!(dat, rand(2), rand(3))
    #=
    skip_index_fn = i -> (i == 4)
    ix = R.add_to_database!(dat, rand(2), rand(3), skip_index_fn)
    @test ix == 5
    =#

    op = C.NonlinearFunction(; func = x -> [x; sum(x)], func_iip=false)
    R.evaluate!(dat, op)
    for j=6:11
        y = dat.y[:, j]
        @test y[1] == y[2] == j
        @test y[3] == 2*j 
    end
    ff = zeros(Bool, 0)
    R.box_search!(ff, dat, [7, 7], [9, 9])
    @assert all( ff[1:6] .== false )
    @assert all( ff[7:9] .== true )
    @assert all( ff[10:11] .== false )
end
#%%

@testset "RBFModel" begin
    op = NonlinearFunction(;
        func = (y, x) -> begin
            y[1] = sum(x)
            for i = Iterators.drop(eachindex(y), 1)
                y[i] = sum(@view(y[1:i-1]))
            end
            nothing
        end,
        func_iip=true
    )
    poly_deg = 1   # TODO 0, nothing
    for kernel in (R.GaussianKernel(), R.InverseMultiQuadricKernel(), R.CubicKernel())
        for dim_x in (1, 2, 4, 10, 50)
        for dim_y in (1, 2, 4, 10)
        for max_points in (dim_x+1, 2*(dim_x + 1))

            Δ = .1
            cfg = R.RBFConfig(; kernel, poly_deg, max_points)
            rbf = C.init_surrogate(cfg, nothing, dim_x, dim_y, nothing, Float64; delta_max = Δ)
            @test val(rbf.params.is_fully_linear_ref) == false
            x0 = rand(dim_x)
            fx0 = zeros(dim_y)
            func_vals!(fx0, op, x0)

            #update_rbf_model!(rbf, op, Δ, x0, fx0)
            C.update!(rbf, op, Δ, x0, fx0, nothing, nothing,)
            @test rbf.params.x0 == x0
            @test val(rbf.params.is_fully_linear_ref) == true
            n_X = val(rbf.params.n_X_ref)
            X = rbf.params.X[:, 1:n_X]
            yi = similar(fx0)
            @test all(X[:, 1] .≈ 0)
            for i=1:n_X-1
                ii = i+1
                xi = X[:, ii]
                @test LA.norm(xi) <= max(cfg.sampling_factor, cfg.search_factor) * Δ
                xi .+= x0
                func_vals!(fx0, op, xi)
                func_vals!(yi, rbf, xi)
                @test isapprox(fx0, yi; rtol=1e-6)
            end 
            rbf0 = deepcopy(rbf)
            #update_rbf_model!(rbf, op, Δ, x0, fx0)
            C.update!(rbf, op, Δ, x0, fx0, nothing, nothing,)
            @test isequal(rbf.params, rbf0.params) # isequal(NaN, NaN) == true
            
            #update_rbf_model!(rbf, op, Δ, x0, fx0; force_rebuild=true)
            C.update!(rbf, op, Δ, x0, fx0, nothing, nothing; force_rebuild=true)
            @test sum(rbf.database.flags_x) == sum(rbf0.database.flags_x)
            @test rbf.params.database_state_ref == rbf0.params.database_state_ref
            n_X = val(rbf.params.n_X_ref)
            X = rbf.params.X[:, 1:n_X]
            yi = similar(fx0)
            for i=1:n_X-1
                ii = i+1
                xi = X[:, ii]
                @test LA.norm(xi) <= max(cfg.sampling_factor, cfg.search_factor) * Δ
                xi .+= x0
                func_vals!(fx0, op, xi)
                func_vals!(yi, rbf, xi)
                @test fx0 ≈ yi
            end 
            for fn in (:n_X_ref, :has_z_new_ref, :is_fully_linear_ref, :z_new, :x0, :delta_ref, 
                :shape_parameter_ref, )
                @test isequal(getfield(rbf.params, fn),  getfield(rbf0.params, fn))
            end

            randx() = rbf.buffers.lb .+ (rbf.buffers.ub .- rbf.buffers.lb) .* rand(dim_x)
            for i = 1:2*n_X
                xi = randx()
                _xi = copy(xi)
                func_vals!(yi, op, xi)
                @test xi == _xi
                add_to_database!(rbf.database, xi, yi)
                @test xi == _xi
            end
            
            #update_rbf_model!(rbf, op, Δ, x0, fx0)
            C.update!(rbf, op, Δ, x0, fx0, nothing, nothing,)

            n_X = val(rbf.params.n_X_ref)
            X = rbf.params.X[:, 1:n_X]
            yi = similar(fx0)
            @test all(X[:, 1] .≈ 0)
    
            for i=1:n_X-1
                ii = i+1
                xi = X[:, ii]
                xi .+= x0
                _xi = copy(xi)
                func_vals!(fx0, op, xi)
                @test xi == _xi
                func_vals!(yi, rbf, xi)
                @test xi == _xi
                isapprox(fx0, yi; rtol=1e-6)
            end

            _rbf = C.init_surrogate(cfg, nothing, dim_x, dim_y, nothing, Real)
            copyto!(_rbf.params, rbf.params)
            copyto!(_rbf.buffers, rbf.buffers)
            Dy = zeros(dim_x, dim_y)
            for _=1:10
                xi = randx()
                jac = FD.jacobian(xi) do x
                    y = zeros(Real, dim_y)
                    func_vals!(y, _rbf, x)
                    y
                end
               
                func_grads!(Dy, rbf, xi)
                @test isapprox(jac, Dy'; rtol=1e-6)
            end
            
        end
        end
        end
    end

end