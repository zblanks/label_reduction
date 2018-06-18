include(joinpath(ARGS[1], "reduce_labels.jl"))
using CSV
using DataFrames
using Combinatorics


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
    t = reshape(t, (size(t)[1], 1))

    # Return the data
    return r, t
end


# Compute how balanced a solution (we're doing sensitivity analysis
# to see how changing the mixing factor affects the solutions)
function compute_balance(z)
    # Compute the worst case balance
    C = size(z)[1]
    L = size(z)[2]
    M = C - L + 1
    W = (M - 1) * (L - 1)

    # Get the set of label combinations
    label_combos = collect(combinations(1:L, 2))
    tau_vals = Array{Float64}(length(label_combos,))
    for i in 1:length(label_combos)
        tau_vals[i] = sum(abs.(z[:, label_combos[i][1]] -
        z[:, label_combos[i][2]]))
    end
    tau_sum = sum(tau_vals)
    return tau_sum / W
end


# Search over all values of k ∈ {2, …, C -1} and get the optimal
# label mapping for each value and we will also investigate how changing
# the mixing factor changes the solution quality
r, t = get_data(ARGS)
C = size(r)[1]
mixing_factors = [1/8, 1/4, 1/2]
n_sample = length(2:(C-1)) * length(mixing_factors)
res = Array{Float64}(n_sample, 6)
count = 1
for L = 2:(C-1)
    for factor in mixing_factors
        # Get our re-shaped version of r and t
        tmp_r, tmp_t = reshape_data(r, t, C, L)

        # Run the optimization problem
        path = joinpath(ARGS[2], "solver_output/mip_output_" * string(L) *
        "_" * string(factor) * ".txt")

        # Solve the MIP and compute the time to get the solution
        tic()
        soln_dict = label_reduce(C, L, tmp_t, tmp_r, factor, path)
        soln_time = toq()
        label_map = convert(DataFrame, soln_dict["map"])

        # Save the map to disk
        CSV.write(joinpath(ARGS[2], "fruit_maps/label_map_" * string(L) * "_"
        * string(factor) * ".csv"), label_map, header=false)

        # Add data to our experiment matrix
        res[count, 1] = L
        res[count, 2] = soln_dict["obj_val"]
        res[count, 3] = factor
        res[count, 4] = compute_balance(soln_dict["map"])
        res[count, 5] = nprocs()
        res[count, 6] = soln_time
        count = count + 1

        # Display our progress
        if L % 5 == 0
            println("Completed $L problems")
        end
    end
end

# Convert the experiment matrix into a DataFrame so that we can save it to disk
df = DataFrame(n_label=res[:, 1], obj_val=res[:, 2], mixing_factor=res[:, 3],
balance_ratio=res[:, 4], n_cpu=res[:, 5], compute_time=res[:, 6])

CSV.write(joinpath(ARGS[2], "sensitivity_analysis", "sensitivity_res.csv"),
df)
