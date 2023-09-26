"""
    pushend!(arr, el)

`push!` element `el` into `arr` and return its index.
"""
function pushend!(arr, el)
    push!(arr, el)
    return lastindex(arr)
end

"""
    cpad(str, len)

Pad the string `str` with white space to have length at least `len`.
The returned string will be center-aligned.
"""
function cpad(str, len)
    l = length(str)
    l >= len && return str
    del = len - l
    p, r = divrem(del, 2)
    p1 = l + p
    p2 = p1 + p + r
    return rpad(lpad(str, p1), p2)
end
