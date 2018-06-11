include(joinpath(ARGS[1], "reduce_labels_ip.jl"))
using CSV
using DataFrames


function get_data(args)
    # Get the script arguments for the path
    r_path = joinpath(args[2], "fruits_sim/class_sim.csv")
    t_path = joinpath(args[2], "fruits_sim/comb_sim.csv")
    r = CSV.read(r_path, header=false)
    t = CSV.read(t_path, header=false)
    return r, t
end


# Get the similarity data
function reshape_data(r, t, C::Int, L::Int)
    # Re-shape it so that it works for our optimiazation problem
    r = reshape(convert(Array, r[:Column1]), C, 1)
    r = repeat(r, inner=(1, L))
    t = convert(Array, t[:Column1])
    t = repeat(t, inner=(1, L))

    # Return the data
    return r, t
end


# Search over all values of k (k ∈ {2, …, C -1}) and get the optimal
# label mapping for each value
r, t = get_data(ARGS)
C = size(r)[1]
for L = 2:(C-1)
    # Get our re-shaped version of r and t
    tmp_r, tmp_t = reshape_data(r, t, C, L)

    # Run the optimization problem
    tmp_path = joinpath(ARGS[2], "solver_output/lp_relax_output_L" * string(L) * ".txt")
    tmp_map = label_reduce(C, L, tmp_t, tmp_r, tmp_path)
    tmp_map = convert(DataFrame, tmp_map)

    # Save the map to disk
    CSV.write(joinpath(ARGS[2], "fruit_maps/linear_label_map_" * string(L) * ".csv"),
    tmp_map, header=false)

    # Display our progress
    if L % 5 == 0
        println("Completed $L problems")
    end
end
