include("reduce_labels_ip.jl")
using CSV

# Define our C and k constants for the script
C = 3
k = 2

# Get the similiarity experiment
r = CSV.read("C:/Users/zqb0731/Documents/label_reduction/experiments/similarity/airport_runway_zoo/class_sim.csv", header=false)
t = CSV.read("C:/Users/zqb0731/Documents/label_reduction/experiments/similarity/airport_runway_zoo/comb_sim.csv", header=false)
r = reshape(convert(Array, r[:Column1]), C, 1)
t = convert(Array, t[:Column1])
t = repeat(t, inner=(1, k))

# Run our IP
println(label_reduce(C, k, t, r))
