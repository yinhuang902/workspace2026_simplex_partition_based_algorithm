bytes = read("single_constraint.txt")
short_str = String(bytes[3:2:min(20000, length(bytes))])
println("Count of -40.0: ", count(i -> occursin("-40", i), split(short_str, ",")))
