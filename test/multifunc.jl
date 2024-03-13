using Compromise
using Test
function multiobjf(x::AbstractVector)
    return [
        sum( (x .- 1).^2 )
        sum( (x .+ 1).^2 )
    ]
end

oop_mat_func_called = Ref(false)
function multiobjf(x::AbstractMatrix)
    global oop_mat_func_called[] = true
    mapreduce(multiobjf, hcat, eachcol(x))
end

mop = MutableMOP(; num_vars=2)
add_objectives!(mop, multiobjf, :rbf; dim_out=2, func_iip=false)

oop_mat_func_called[] = false
ret = optimize(mop, [π, -ℯ])
xopt = opt_vars(ret)
@test isapprox(xopt[1], xopt[2]; rtol=1e-4)
@test !oop_mat_func_called[]
#%%
mop = MutableMOP(; num_vars=2)
add_objectives!(mop, multiobjf, :rbf; dim_out=2, func_iip=false, chunk_size=Inf)

oop_mat_func_called[] = false
ret = optimize(mop, [π, -ℯ])
xopt = opt_vars(ret)
@test isapprox(xopt[1], xopt[2]; rtol=1e-4)
@test oop_mat_func_called[]

#%%
function multiobjf(y::AbstractVector, x::AbstractVector)
    y[1] = sum( (x .- 1).^2 )
    y[2] = sum( (x .+ 1).^2 )
    nothing
end

iip_mat_func_called = Ref(false)
iip_sx = Ref(0)
function multiobjf(y::AbstractMatrix, x::AbstractMatrix) 
    global iip_mat_func_called[] = true
    global iip_sx
    iip_sx[] = max(size(x, 2), iip_sx[])
    map(multiobjf, eachcol(y), eachcol(x))
    nothing
end

mop = MutableMOP(; num_vars=2)
add_objectives!(mop, multiobjf, :rbf; dim_out=2, func_iip=true)

iip_mat_func_called[] = false
ret = optimize(mop, [π, -ℯ])
xopt = opt_vars(ret)
@test isapprox(xopt[1], xopt[2]; rtol=1e-4)
@test !iip_mat_func_called[]
#%%
mop = MutableMOP(; num_vars=2)
add_objectives!(mop, multiobjf, :rbf; dim_out=2, func_iip=true, chunk_size=Inf)

iip_mat_func_called[] = false
ret = optimize(mop, [π, -ℯ]);
xopt = opt_vars(ret)
@test isapprox(xopt[1], xopt[2]; rtol=1e-4)
iip_mat_func_called[]

#%%
was_parallel = Ref(false)
function parallel_objf(x)
    if x isa Vector
        return multiobjf(x)
    else
        n_x = size(x, 2)
        Y = zeros(2, n_x)
        Y_lock = ReentrantLock()
        Threads.@threads for i=1:n_x
            y = multiobjf(@view(x[:, i]))
            lock(Y_lock) do
                global was_parallel[] = true
                Y[:, i] .= y
            end
        end
        return Y
    end
end
#%%
mop = MutableMOP(; num_vars=2)
add_objectives!(mop, parallel_objf, :rbf; dim_out=2, func_iip=false, chunk_size=Inf)

was_parallel[] = false
ret = optimize(mop, [π, -ℯ]);
xopt = opt_vars(ret)
@test isapprox(xopt[1], xopt[2]; rtol=1e-4)
@test was_parallel[] = true
#%%
mop = MutableMOP(; num_vars=2, lb = [-4.0, -4.0], ub = [4.0, 4.0])

add_objectives!(mop, multiobjf, :rbf; dim_out=2, func_iip=true, chunk_size=Inf)

iip_sx[] = 0
iip_mat_func_called[] = false
ret = optimize(mop, [π, -ℯ]);
xopt = opt_vars(ret)
@test isapprox(xopt[1], xopt[2]; rtol=1e-4)
@test iip_mat_func_called[]
@test iip_sx[] <= 3
#%%
mop = MutableMOP(; num_vars=2, lb = [-4.0, -4.0], ub = [4.0, 4.0])

add_objectives!(mop, multiobjf, :rbf; dim_out=2, func_iip=false, chunk_size=Inf)

oop_mat_func_called[] = false
ret = optimize(mop, [π, -ℯ]);
xopt = opt_vars(ret)
@test isapprox(xopt[1], xopt[2]; rtol=1e-4)
@test oop_mat_func_called[]