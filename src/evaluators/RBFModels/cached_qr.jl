# This file contains utilities to pre-cache working arrays for repeated QR decomposition.
# It is inspired by FastLapackInterface.
# Once https://github.com/DynareJulia/FastLapackInterface.jl/issues/40 is resolved
# we can think about using that package again.
import LinearAlgebra: BlasInt, BlasFloat, require_one_based_indexing, chkstride1 
import LinearAlgebra.BLAS: @blasfunc
import LinearAlgebra.LAPACK: chklapackerror, libblastrampoline
import Libdl: dlsym

import StridedViews: sview

mutable struct QRWYWs{
    R<:Number,
    MT<:StridedMatrix{R},
}
    work::Vector{R}
    T::MT
end

function QRWYWs(A::StridedMatrix{T}; kwargs...) where {T <: BlasFloat}
    resize!(
        QRWYWs(T[], Matrix{T}(undef, 0, 0)), 
        A; 
        kwargs...
    )
end

function Base.resize!(
    ws::QRWYWs, A::StridedMatrix; 
    blocksize=36, work=true
)
    require_one_based_indexing(A)
    chkstride1(A)
    m, n = BlasInt.(size(A))
    minmn = min(m, n)
    nb = min(minmn, blocksize)

    ws.T = similar(ws.T, nb, minmn)
    if work
        resize!(ws.work, nb*n)
    end
    return ws
end

for (elty, geqrt) in (
    (:Float64, :dgeqrt_),
    (:Float32, :sgeqrt_),
    (:ComplexF64, :zgeqrt_),
    (:ComplexF32, :cgeqrt_),
)
    @eval function geqrt!(ws::QRWYWs, A::AbstractMatrix{$(elty)}; resize=true, blocksize=36)
        require_one_based_indexing(A)
        chkstride1(A)
        m, n = size(A)
        minmn = min(m, n)
        nb = min(minmn, blocksize)
        t1 = size(ws.T, 1)
        if t1 < nb
            if resize
                resize!(ws, A; blocksize, work=true)
            else
                #throw(WorkspaceSizeError(nb, minmn))
                error("Cannot resize.")
            end
        end

        T = @view(ws.T[1:nb, 1:minmn])
        if minmn > 0
            lda = max(1, stride(A, 2))
            work = @view(ws.work[1:nb*n])
            info = Ref{BlasInt}()            
            ccall((@blasfunc($geqrt), libblastrampoline), Cvoid,
                (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                Ptr{BlasInt}),
                m, n, nb, A,
                lda, T, max(1,stride(T,2)), work,
                info)
            chklapackerror(info[])
        end
        return A, T
    end
end
#=
 function geqrt!(A::AbstractMatrix{$elty}, T::AbstractMatrix{$elty})
            require_one_based_indexing(A, T)
            chkstride1(A)
            m, n = size(A)
            minmn = min(m, n)
            nb = size(T, 1)
            if nb > minmn
                throw(ArgumentError("block size $nb > $minmn too large"))
            end
            lda = max(1, stride(A,2))
            work = Vector{$elty}(undef, nb*n)
            if n > 0
                info = Ref{BlasInt}()
                ccall((@blasfunc($geqrt), libblastrampoline), Cvoid,
                    (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                     Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                     Ptr{BlasInt}),
                     m, n, nb, A,
                     lda, T, max(1,stride(T,2)), work,
                     info)
                chklapackerror(info[])
            end
            A, T
        end
        =#