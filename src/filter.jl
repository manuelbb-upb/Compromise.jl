@with_kw struct StandardFilter{T}
	gamma :: T = DEFAULT_FLOAT_TYPE(0.1) # envelope factor
	theta_vec :: Vector{T} = T[]        # stores (1-γ)θᵢ
	phi_vec :: Vector{T} = T[]          # stores Φᵢ - γθᵢ

    min_phi :: Base.RefValue{T} = Ref(T(Inf))
    min_theta :: Base.RefValue{T} = Ref(T(Inf))
end

num_entries(filter::StandardFilter) = length(filter.theta_vec)
function add_to_filter!(filter::StandardFilter, θ, Φ)
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

    if _θ < filter.min_theta[]
        filter.min_theta[] = _θ
    end
    if _Φ < filter.min_phi[]
        filter.min_phi[] = _Φ
    end
    return num_entries(filter)
end

function is_acceptable(filter::StandardFilter, θ, Φ)
    θmin = filter.min_theta[]
    Φmin = filter.min_phi[]
    if θ <= θmin 
        return true
    end
    if Φ <= Φmin
        return true
    end

    n_entries = num_entries(filter)
    for j=1:n_entries
        θj = filter.theta_vec[j]
        Φj = filter.phi_vec[j]
        if θ > θj && Φ > Φj
            return false
        end
    end
    return true
end

function is_acceptable(filter::StandardFilter, θ_test, Φ_test, θ_add, Φ_add)
    offset = θ_add * filter.gamma
    θj = θ_add - offset
    Φj = Φ_add - offset
    if θ_test <= θj || Φ_test <= Φj
        return is_acceptable(filter, θ_test, Φ_test)
    else
        return false
    end
end