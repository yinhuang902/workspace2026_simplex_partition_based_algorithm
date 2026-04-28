bytes = read("single_constraint_2.txt")
short_str = String(bytes[3:2:min(20000, length(bytes))])
println(short_str)
