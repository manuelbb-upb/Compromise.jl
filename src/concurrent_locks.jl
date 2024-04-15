lock_write(::AbstractReadWriteLock)=nothing
lock_write(@nospecialize(func), ::AbstractReadWriteLock)=func()
unlock_write(::AbstractReadWriteLock)=nothing
lock_read(::AbstractReadWriteLock)=nothing
lock_read(@nospecialize(func), ::AbstractReadWriteLock)=func()
unlock_read(::AbstractReadWriteLock)=nothing

function init_rw_lock(rwtype)
    error("'init_rw_lock` not defined for $(rwtype)")
end

struct PseudoRWLock <: AbstractReadWriteLock end

function init_rw_lock(rwtype::Type{PseudoRWLock})
    return PseudoRWLock()
end

function default_rw_lock()
    return init_rw_lock(PseudoRWLock)
end