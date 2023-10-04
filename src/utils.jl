function vec2str(x, max_entries=typemax(Int), digits=10)
	x_end = min(length(x), max_entries)
	_x = x[1:x_end]
	_, i = findmax(abs.(_x))
	len = length(string(trunc(_x[i]))) + digits + 1
	x_str = "[\n"
	for xi in _x
		x_str *= "\t" * lpad(string(round(xi; digits)), len, " ") * ",\n"
	end
	x_str *= "]"
	return x_str
end

promote_modulo_nothing(T1, ::Type{Nothing})=T1
promote_modulo_nothing(T1, T2)=Base.promote_type(T1, eltype(T2))
macro serve(ex)
	ret_val = gensym()
	return quote
		$(ret_val) = $(ex)
		if !isnothing($(ret_val))
			return $(ret_val)
		end
	end |> esc
end

macro exit(ex)
	ret_val = gensym()
	return quote
		$(ret_val) = $(ex)
		if !isnothing($(ret_val))
			break
		end
	end |> esc
end