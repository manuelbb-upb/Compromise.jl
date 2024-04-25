# ### Sample Database
Base.@kwdef struct RBFDatabase{T<:Number, RWLType<:AbstractReadWriteLock}
    dim_x :: Int
    dim_y :: Int

    max_size :: Int
    chunk_size :: Int
    x :: ElasticArray{T, 2}
    y :: ElasticArray{T, 2}
    flags_x :: Vector{Int}
    flags_y :: Vector{Int}

    current_size :: Base.RefValue{Int}
    state :: MutableNumber{UInt64}

    rwlock :: RWLType = default_rw_lock()
end

function db_current_size(db)
    @unpack current_size, rwlock = db
    #lock_read(rwlock) do
        current_size[]
    #end
end

function db_state(db)
    @unpack state, rwlock = db
    #lock_read(rwlock) do
        val(state)
    #end
end

function db_sync_flags_x!(flags, db)
    @unpack flags_x, rwlock = db
    #lock_read(rwlock) do
        for (i, s) = enumerate(flags_x)
            flags[i] = s > 0
        end
    #end
end

function db_view_x(db::RBFDatabase{T}) where T
    return db.x :: ElasticArray{T, 2, 1, Vector{T}}
end

function db_view_y(db::RBFDatabase{T}) where T
    return db.y :: ElasticArray{T, 2, 1, Vector{T}}
end

function filtered_view_x(db::RBFDatabase, flags)
    base_view = db_view_x(db)
    n = size(base_view, 2)
    return view(base_view, 1:db.dim_x, view(flags, 1:n))
end
function filtered_view_y(db::RBFDatabase, flags)
    base_view = db_view_y(db)
    n = size(base_view, 2)
    return view(base_view, 1:db.dim_y, view(flags, 1:n))
end

function init_rbf_database(
    rbf_cfg, dim_x, dim_y, T=DEFAULT_FLOAT_TYPE, 
    rwlock::AbstractReadWriteLock=default_rw_lock()
)
    @unpack database_size, database_chunk_size = rbf_cfg
    return init_rbf_database(dim_x, dim_y, database_size, database_chunk_size, T)
end

function init_rbf_database(
    dim_x::Integer, dim_y::Integer, 
    database_size::Union{Nothing,Integer},
    database_chunk_size::Union{Nothing, Integer}, 
    ::Type{T}=DEFAULT_FLOAT_TYPE,
    rwlock::Union{Nothing,AbstractReadWriteLock}=nothing
) where {T<:Number}
    min_points = dim_x + 1 :: Integer

    ## some heuristics to initialize database of points
    max_size = if isnothing(database_size)
        ## 1 GB = 1 billion bytes = 10^9 bytes
        ## byte-size of one colum: `sizeof(T)*dim_x`
        ## 1 GB database => 10^9/(sizeof(T)*dim_x)
        if isconcretetype(T)
            max(2 * min_points, round(Int, 10^9/(sizeof(T)*dim_x)))
        else
            2 * min_points
        end
    else
        database_size
        #=if database_size < min_points
            @warn "There is not enough storage in the database, so we are using $(min_points) columns."
            min_points
        else
            database_size
        end=#
    end

    chunk_size = if isnothing(database_chunk_size)
        min(2*min_points, max_size)
    else
        max(1, min(database_chunk_size, max_size))
    end

    x = ElasticArray{T, 2}(undef, dim_x, chunk_size)
    y = ElasticArray{T, 2}(undef, dim_y, chunk_size)

    flags_x = zeros(Int, chunk_size)
    flags_y = zeros(Int, chunk_size)

    state = MutableNumber(zero(UInt64))
    if isnothing(rwlock)
        rwlock = default_rw_lock()
    end
    current_size = Ref(chunk_size)
    
    return RBFDatabase(;
        dim_x, dim_y, max_size, chunk_size, x, y, 
        flags_x, flags_y, state, rwlock, current_size
    )   
end

function db_grow!(rbf_database; force_chunk=0)
    @unpack x, y, flags_x, flags_y, rwlock = rbf_database
    @unpack dim_x, dim_y, chunk_size, max_size, current_size = rbf_database

    n_x = size(x, 2)
    #=n_x = lock_read(rwlock) do
        size(x, 2)
    end
    _dim_y, n_y = size(y)
    @assert n_x == n_y
    @assert n_x == length(flags_x)
    @assert n_y == length(flags_y)
    =#

    free_slots = max_size - n_x
    if free_slots > 0 || force_chunk > 0
        new_chunk = force_chunk > 0 ? force_chunk : max(min(chunk_size, free_slots), 1)
        new_size = n_x + new_chunk
        #lock(rwlock) do
            resize!(x, dim_x, new_size)
            resize!(y, dim_y, new_size)
            append_zeros!(flags_x, new_chunk)
            append_zeros!(flags_y, new_chunk)
            current_size[] = new_size
        #end
        return true
    end
    return false
end

function append_zeros!(flag_vec::AbstractVector{T}, chunk_size) where T
    append!(flag_vec, zeros(T, chunk_size))
end

function add_to_database!(
    rbf_database::RBFDatabase, x::AbstractVector, 
    #@nospecialize(skip_index_fn = i -> false)
    ;
    force_new::Bool=false,
)
    @unpack dim_x, flags_x, rwlock = rbf_database
    @assert dim_x == length(x)
    
    #lock_read(rwlock)
    
    x_index = 0
    min_flag = typemax(Int)
    oldest_index = 0
    for i = eachindex(flags_x)
        #skip_index_fn(i) && continue
        flag = flags_x[i]
        if flag < min_flag 
            min_flag = flag
            oldest_index = i
        end
        flag > 0 && continue
        
        x_index = i
        break 
    end
    n_x = length(flags_x)
    
    #unlock_read(rwlock)

    ## If we did not find a free slot, try to grow database
    if x_index == 0
        grow_db_success = db_grow!(rbf_database)
        if grow_db_success
           x_index = n_x + 1
           min_flag = 0
        end
    end
    
    if x_index == 0
        x_index = oldest_index
    end

    if x_index == 0 && force_new
        grow_db_success = db_grow!(rbf_database; force_chunk=1)
        @assert grow_db_success
        x_index = n_x + 1
        min_flag = 0
    end

    return db_set_x!(rbf_database, x, x_index, min_flag + 1)
end

function add_to_database!(
    rbf_database::RBFDatabase, x::AbstractVector, fx::AbstractVector
    #@nospecialize(skip_index_fn = i -> false)
    ;
    force_new::Bool=false, 
)
    @unpack dim_y = rbf_database
    @assert length(fx) == dim_y
    #x_index = add_to_database!(rbf_database, x, skip_index_fn; force_new)
    x_index = add_to_database!(rbf_database, x; force_new)
    return db_set_y!(rbf_database, fx, x_index)
end

function db_set_x!(
    rbf_database, x, x_index, flag=1
)
    @unpack flags_x, state = rbf_database
    X = rbf_database.x
    return db_set_x!(X, flags_x, state, x, x_index, flag)
end

function db_set_x!(X, flags_x, state, x, x_index, flag)
    if x_index > 0
        #lock(rwlock) do
        X[:, x_index] .= x
        flags_x[x_index] = flag
        _state = val(state)
        val!(state, hash(x, _state))
    #end
    end
    return x_index
end

function db_set_y!(rbf_database, fx, x_index)
    @unpack y, flags_y, flags_x, state = rbf_database
    return db_set_y!(y, flags_y, flags_x, state, fx, x_index)
end
function db_set_y!(Y, flags_y, flags_x, state, fx, x_index)
    if x_index > 0
        #lock(rwlock) do
        Y[:, x_index] .= fx
        flags_y[x_index] = flags_x[x_index]
        _state = val(state)
        val!(state, hash(fx, _state))
    #end
    end
    return x_index
end

function evaluate!(rbf_database, op)
    @unpack flags_x, flags_y, rwlock, state = rbf_database
   
    ## find those entries, where `x` is set, but `y` is missing
    ## TODO in case of eval-parallelization, get rid of the for loop(s)
    #lock(rwlock) do
        X = rbf_database.x
        Y = rbf_database.y
        for (i, xset) = enumerate(flags_x)
            if xset > 0
                if flags_y[i] < flags_x[i]
                    x = @view(X[:, i])
                    y = @view(Y[:, i])
                    @ignoraise func_vals!(y, op, x)
                    
                    _state = val(state)
                    val!(state, hash(y, _state))
                    
                    flags_y[i] = flags_x[i]
                end
            end
        end
    #end
    return nothing
end

@views function box_search!(
    filter_flags::AbstractVector{Bool}, rbf_database::RBFDatabase{T}, lb, ub; 
    needs_y=true, xor=false # `xor==true`, then set `filter_flags[i]=false` if `filter_flags[i]==true` and it would be marked again
) where T
    @unpack flags_x, flags_y, rwlock, current_size = rbf_database
    
    # lock_read(rwlock)
    
    cs = current_size[]
    if length(filter_flags) <= cs
        resize!(filter_flags, cs)
    end
    X = db_view_x(rbf_database)
    @unpack dim_x = rbf_database
    for (i, flag) = enumerate(flags_x)
        fi = filter_flags[i] :: Bool
        filter_flags[i] = false
        flag <= 0 && continue
        if needs_y && flags_y[i] != flag
            continue
        end
        x_in_bounds = true
        for ix = 1:dim_x
            xi = X[ix, i]::T
            if lb[ix] > xi || ub[ix] < xi
                x_in_bounds = false
                break
            end
        end
        filter_flags[i] = if xor && fi && x_in_bounds
            false
        else
            x_in_bounds
        end
    end
    
    # unlock_read(rwlock)

    return nothing        
end

