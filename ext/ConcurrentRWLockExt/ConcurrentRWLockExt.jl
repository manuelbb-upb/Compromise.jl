module ConcurrentRWLockExt

import Compromise as C
import Compromise: AbstractReadWriteLock, lock_write, unlock_write, lock_read, unlock_read, init_rw_lock

if isdefined(Base, :get_extension)
    import ConcurrentUtils as CU
    import ConcurrentUtils: ReadWriteLock
else
    import ..ConcurrentUtils as CU
    import ..ConcurrentUtils: ReadWriteLock
end

struct ConcurrentRWLock <: AbstractReadWriteLock
    wrapped :: ReadWriteLock
end
ConcurrentRWLock() = ConcurrentRWLock(ReadWriteLock())

lock_read(l::ConcurrentRWLock)=CU.lock_read(l.wrapped)
lock_read(@nospecialize(f), l::ConcurrentRWLock)=CU.lock_read(f, l.wrapped)
unlock_read(l::ConcurrentRWLock)=CU.unlock_read(l.wrapped)

lock_write(l::ConcurrentRWLock)=lock(l.wrapped)
lock_write(@nospecialize(f), l::ConcurrentRWLock)=lock(f, l.wrapped)
unlock_write(l::ConcurrentRWLock)=unlock(l.wrapped)

init_rw_lock(::Type{ConcurrentRWLock})=ConcurrentRWLock(ReadWriteLock())
init_rw_lock(::Type{ReadWriteLock})=ConcurrentRWLock(ReadWriteLock())

end