bytes = read("single_constraint_2.txt")
short_str = String(bytes[3:2:min(20000, length(bytes))])
parts = split(short_str, "coeffs: [")
if length(parts) > 1
    coeffs_str = split(parts[2], "]")[1]
    # some elements might be chopped, e.g. " -40.0" or " -" if string was truncated
    raw_strs = filter(x -> length(x) > 0 && !(x == " " || x == "" || x == "-"), strip.(split(coeffs_str, ",")))
    coeffs = [parse(Float64, x) for x in raw_strs if tryparse(Float64, x) !== nothing]
    
    # count exact -40.0
    c_40 = count(x -> x == -40.0, coeffs)
    println("Exact -40.0: ", c_40)
end
