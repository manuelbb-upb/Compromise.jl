# ### Sample Database

Base.@kwdef struct RBFDatabase{T<:Number}
    dim_x :: Int
    dim_y :: Int

    max_size :: Int
    chunk_size :: Int
    database_x :: ElasticArray{T, 2}
    database_y :: ElasticArray{T, 2}
    flags_x :: Vector{Int}
    flags_y :: Vector{Int}

    filter_flags :: Vector{Bool} 
    state :: MutableNumber{UInt64}
end

function filter_set!(db)
    for (i, s) = enumerate(db.flags_x)
        db.filter_flags[i] = s > 0
    end
end

function entries_x(db::RBFDatabase{T}) where T
    return db.database_x :: ElasticArray{T, 2, 1, Vector{T}}
end
function entries_y(db::RBFDatabase{T}) where T
    return db.database_y :: ElasticArray{T, 2, 1, Vector{T}}
end
function filtered_view_x(db::RBFDatabase)
    return view(entries_x(db), 1:db.dim_x, db.filter_flags)
end
function filtered_view_y(db::RBFDatabase)
    return view(entries_y(db), :, db.filter_flags)
end

function init_rbf_database(rbf_cfg, dim_x, dim_y, T=DEFAULT_FLOAT_TYPE)
    @unpack database_size, database_chunk_size = rbf_cfg
    return init_rbf_database(dim_x, dim_y, database_size, database_chunk_size, T)
end

function init_rbf_database(
    dim_x::Integer, dim_y::Integer, 
    database_size::Union{Nothing,Integer},
    database_chunk_size::Union{Nothing, Integer}, 
    ::Type{T}=DEFAULT_FLOAT_TYPE
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

    database_x = ElasticArray{T, 2}(undef, dim_x, chunk_size)
    database_y = ElasticArray{T, 2}(undef, dim_y, chunk_size)

    flags_x = zeros(Int, chunk_size)
    flags_y = zeros(Int, chunk_size)

    filter_flags = zeros(Bool, chunk_size)

    state = MutableNumber(zero(UInt64))
    return RBFDatabase(;
        dim_x, dim_y, max_size, chunk_size, database_x, database_y, 
        flags_x, flags_y, filter_flags, state
    )   
end

function grow_database!(rbf_database, force_chunk=0)
    @unpack database_x, database_y, flags_x, flags_y, filter_flags = rbf_database
    @unpack dim_x, dim_y, chunk_size, max_size = rbf_database
   
    _dim_x, n_x = size(database_x)
    _dim_y, n_y = size(database_y)
    @assert n_x == n_y
    @assert n_x == length(flags_x)
    @assert n_y == length(flags_y)

    free_slots = max_size - n_x
    if free_slots > 0 || force_chunk > 0
        new_chunk = force_chunk > 0 ? force_chunk : max(min(chunk_size, free_slots), 1)
        new_size = n_x + new_chunk 
        resize!(database_x, dim_x, new_size)
        resize!(database_y, dim_y, new_size)
        append_zeros!(flags_x, new_chunk)
        append_zeros!(flags_y, new_chunk)
        append_zeros!(filter_flags, new_chunk)
        return true
    end
    return false
end

function append_zeros!(flag_vec, chunk_size)
    append!(flag_vec, zeros(eltype(flag_vec), chunk_size))
end

function add_to_database!(
    rbf_database::RBFDatabase, x::AbstractVector, @nospecialize(skip_index_fn = i -> false);
    force_new::Bool=false
)
    @unpack dim_x, flags_x = rbf_database
    @assert dim_x == length(x)
    n_x = length(flags_x)
    
    x_index = 0
    min_flag = typemax(Int)
    min_flag_i = 0
    for i = eachindex(flags_x)
        skip_index_fn(i) && continue
        flag = flags_x[i]
        if flag < min_flag 
            min_flag = flag
            min_flag_i = i
        end
        flag > 0 && continue
        
        x_index = i
        break 
    end

    ## If we did not find a free slot, try to grow database
    if x_index == 0
        grow_db_success = grow_database!(rbf_database)
        if grow_db_success
           x_index = n_x + 1
           @assert flags_x[x_index] == 0
        end
    end
    
    if x_index == 0
        x_index = min_flag_i
    end

    if x_index == 0 && force_new
        grow_db_success = grow_database!(rbf_database)
        @assert grow_db_success
        x_index = n_x + 1
        @assert flags_x[x_index] == 0
    end

    return set_x!(rbf_database, x, x_index, min_flag + 1)
end

function add_to_database!(
    rbf_database::RBFDatabase, x::AbstractVector, fx::AbstractVector, 
    @nospecialize(skip_index_fn = i -> false)
)
    @unpack dim_y = rbf_database
    @assert length(fx) == dim_y
    x_index = add_to_database!(rbf_database, x, skip_index_fn)
    return set_y!(rbf_database, fx, x_index)
end

function set_x!(rbf_database, x, x_index, flag=1)
    @unpack database_x, flags_x, state = rbf_database
    if x_index != 0
        database_x[:, x_index] .= x
        flags_x[x_index] = flag
        _state = val(state)
        val!(state, hash(x, _state))
    end
    return x_index
end

function set_y!(rbf_database, fx, x_index)
    @unpack database_y, flags_y, flags_x, state = rbf_database
    if x_index != 0
        database_y[:, x_index] .= fx
        flags_y[x_index] = flags_x[x_index]
        _state = val(state)
        val!(state, hash(fx, _state))
    end
    return x_index
end

function evaluate!(rbf_database, op)
    @unpack database_x, database_y, flags_x, flags_y, state = rbf_database
   
    ## find those entries, where `x` is set, but `y` is missing
    ## TODO in case of parallelization, get rid of the for loop(s)
    for (i, xset) = enumerate(flags_x)
        if xset > 0
            if flags_y[i] < flags_x[i]
                x = @view(database_x[:, i])
                y = @view(database_y[:, i])
                @ignoraise func_vals!(y, op, x)
                
                _state = val(state)
                val!(state, hash(y, _state))
                
                flags_y[i] = flags_x[i]
            end
        end
    end
    return nothing
end

@views function box_search!(rbf_database::RBFDatabase{T}, lb, ub; needs_y=true) where T
    @unpack flags_x, flags_y, filter_flags = rbf_database
    database_x = entries_x(rbf_database)
    dim_x = size(database_x, 1)::Int
    for (i, flag) = enumerate(flags_x)
        filter_flags[i] = false
        flag <= 0 && continue
        if needs_y && flags_y[i] == 0
            continue
        end
        x_in_bounds = true
        for ix = 1:dim_x
            xi = database_x[ix, i]::T
            if lb[ix] > xi || ub[ix] < xi
                x_in_bounds = false
                break
            end
        end
        filter_flags[i] = x_in_bounds 
    end
    return nothing        
end

