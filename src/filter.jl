@with_kw struct WeakFilter{T}
	gamma :: T = DEFAULT_PRECISION(0.1) # envelope factor
	theta_vec :: Vector{T} = T[]        # stores (1-γ)θᵢ
	phi_vec :: Vector{T} = T[]          # stores Φᵢ - γθᵢ
end

num_entries(filter::WeakFilter) = length(filter.theta_vec)
function add_to_filter!(filter::WeakFilter, θ, Φ)
    γ = filter.gamma
    
    offset = γ * θ
    _θ = θ - offset
    _Φ = Φ - offset

    n_entries = num_entries(filter)
    delete_entry = zeros(Bool, n_entries)
    for j=1:n_entries
        θj = filter.theta_vec[j]
        Φj = filter.phi_vec[j]
        if θj >= _θ && Φj >= _Φ
            delete_entry[j] = true
        end
    end
    deleteat!(filter.theta_vec,delete_entry)
    deleteat!(filter.phi_vec,delete_entry)
    push!(filter.theta_vec, _θ)
    push!(filter.phi_vec, _Φ)
    return num_entries(filter)
end

function is_acceptable(filter::WeakFilter, θ, Φ)
    n_entries = num_entries(filter)
    for j=1:n_entries
        θj = filter.theta_vec[j]
        Φj = filter.phi_vec[j]
        if θ > θj && Φj > Φ
            return false
        end
    end
    return true
end

function is_acceptable(filter::WeakFilter, θ_test, Φ_test, θ_add, Φ_add)
    offset = θ_add * filter.gamma
    θj = θ_add - offset
    Φj = Φ_add - offset
    if θ_test <= θj || Φ_test <= Φj
        return is_acceptable(filter, θ_test, Φ_test)
    else
        return false
    end
end