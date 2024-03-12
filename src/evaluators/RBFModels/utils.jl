# Trust region boundaries are computed for the unshifted
# coordinate system, i.e., they can be used to query 
# the database.
# On the other hand, if we inspect “directions” as new 
# points, we might have to add the trust region center 
# `x` to shift them into these boxes.
 function do_qr!(::Any, A, _A=nothing)
    return LA.qr(A)
end

function do_qr!(ws::QRWYWs, A, _A=similar(A))
    m, n = size(A)
    __A = @view(_A[1:m, 1:n])
    copyto!(__A, A)
    return LA.QRCompactWY(geqrt!(ws, __A)...)
end

function _q_factor(F)
    return LA.QRCompactWYQ(getfield(F, :factors), F.T)
end
function _r_factor(F)    
    m, n = size(F)
    return LA.triu!(getfield(F, :factors)[1:min(m,n), 1:n])
end

function qr!(Q, R, A, ws::QRWYWs, _A=similar(A))
    qr = do_qr!(ws, A, _A)
    copyto!(Q, _q_factor(qr))
    copyto!(R, _r_factor(qr))
    nothing
end

function qr!(Q, R, A, ws::Nothing, _A=nothing)
    q, r = LA.qr(A)
    copyto!(Q, q)
    copyto!(R, r)        
end