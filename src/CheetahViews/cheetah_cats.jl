import PaddedViews: PaddedView

function cheetah_vcat(slices...)
    return CheetahView(slices, Val(1))
end

function cheetah_hcat(slices...)
    return CheetahView(slices, Val(2))
end

function cheetah_blockcat(A::AbstractMatrix, B::AbstractMatrix)
    mA, nA = size(A)
    mB, nB = size(B)
    n = nA + nB
    _A = PaddedView(0, A, (mA, n))
    _B = PaddedView(0, B, (mB, n), (1, nA+1))

    return cheetah_vcat(_A, _B)
end
