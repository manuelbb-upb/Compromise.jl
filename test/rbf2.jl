using Test

import Compromise.RBFModels: unfilter_index!

#%%
filter_flags = ones(Bool, 10)
chosen_index = collect(1:10)
_chosen_index = copy(chosen_index)
unfilter_index!(chosen_index, filter_flags; ix1=1)
@test chosen_index[1:10] == findall(filter_flags)[_chosen_index[1:10]]
@test all(filter_flags[chosen_index[chosen_index .> 0]])

filter_flags = ones(Bool, 10)
chosen_index = collect(1:10)
_chosen_index = copy(chosen_index)
unfilter_index!(chosen_index, filter_flags; ix1=5)
@test chosen_index[5:10] == findall(filter_flags)[_chosen_index[5:10]]
@test all(filter_flags[chosen_index[chosen_index .> 0]])

filter_flags = ones(Bool, 10)
filter_flags[1] = false
chosen_index = collect(1:10)
_chosen_index = copy(chosen_index)
unfilter_index!(chosen_index, filter_flags; ix1=1)
@test chosen_index == [2, 3, 4, 5, 6, 7, 8, 9, 10, -1]
@test chosen_index[1:9] == findall(filter_flags)[_chosen_index[1:9]]    # 1:9 because findall(...) is 9-elements long
@test all(filter_flags[chosen_index[chosen_index .> 0]])

filter_flags = ones(Bool, 10)
filter_flags[2] = false
chosen_index = collect(1:10)
unfilter_index!(chosen_index, filter_flags; ix1=1)
@test chosen_index == [1, 3, 4, 5, 6, 7, 8, 9, 10, -1]
@test all(filter_flags[chosen_index[chosen_index .> 0]])

filter_flags = ones(Bool, 10)
filter_flags[1] = false
filter_flags[3] = false
chosen_index = collect(1:10)
unfilter_index!(chosen_index, filter_flags; ix1=1)
@test chosen_index == [2, 4, 5, 6, 7, 8, 9, 10, -1, -1]
@test all(filter_flags[chosen_index[chosen_index .> 0]])

filter_flags = zeros(Bool, 10)
chosen_index = collect(1:10)
unfilter_index!(chosen_index, filter_flags; ix1=1)
@test all( chosen_index .== -1 )

filter_flags = ones(Bool, 10)
filter_flags[1] = false
filter_flags[3] = false
chosen_index = collect(1:10)
unfilter_index!(chosen_index, filter_flags;)
# because ix1=2 by default, first entry is not changed:
@test chosen_index == [1, 4, 5, 6, 7, 8, 9, 10, -1, -1]

filter_flags = ones(Bool, 10)
filter_flags[1] = false
filter_flags[3] = false
chosen_index = collect(1:10)
unfilter_index!(chosen_index, filter_flags; ix2=9)
# because ix1=2 by default, first entry is not changed:
@test chosen_index == [1, 4, 5, 6, 7, 8, 9, 10, -1, 10]

filter_flags = ones(Bool, 10)
filter_flags[1] = false
filter_flags[3] = false
chosen_index = collect(1:10)
unfilter_index!(chosen_index, filter_flags; ix1=3, ix2=8)
# because ix1=2 by default, first entry is not changed:
@test chosen_index == [1, 2, 5, 6, 7, 8, 9, 10, 9, 10]
#%%

import Compromise.RBFModels: sort_with_flags!, postsort_flags!, presort_flags!

dX = 10
dY = 3
min_points = dX + 1
max_points = 2*min_points

X = rand(dX, max_points)
Y = rand(dY, max_points)

_X = copy(X)
_Y = copy(Y)
 
xtmp = rand(dX)
ytmp = rand(dY)

db_index = fill(-1, min_points)
sorting_flags = rand(Bool, min_points)

for istart = 1:min_points
    for n_X = istart:min_points
        
        db_index[istart:n_X] .= rand(Int, n_X - istart + 1)
        presort_flags!(sorting_flags, db_index, istart, n_X)

        for j = istart:n_X
            if !sorting_flags[j]
                X[:, j] .= 0
                Y[:, j] .= 0
            end
        end

        N = sort_with_flags!(X, Y, db_index, sorting_flags, xtmp, ytmp, istart, n_X)
        
        @test N - istart + 1 == sum(db_index[istart:n_X] .> 0)
        @test all(db_index[istart:N] .> 0)
        @test all(db_index[N+1:n_X] .< 1)
        @test abs(sum(X[:, istart:N])) < 1e-10
        @test abs(sum(Y[:, istart:N])) < 1e-10
        
        if N < n_X
            X[1, N+1] = Inf
            Y[1, N+1] = NaN
            postsort_flags!(sorting_flags, Y, N+1, n_X)
            @test sorting_flags[N+1] == true
            @test sum(sorting_flags[N+1:n_X]) == 1 

            M = sort_with_flags!(X, Y, db_index, sorting_flags, xtmp, ytmp, N+1, n_X)
            @test M == n_X - 1
            @test isinf(X[1, M+1])
            @test isnan(Y[1, M+1])
        end
        
        db_index .= -1
        X .= _X
        Y .= _Y
    end
end