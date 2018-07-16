__precompile__()

module LocalSearch
using Combinatorics
export local_search


"""
Generates a neighbor from the current best solution
"""
function gen_neighbor(z::Array{Int64, 2}, move_dict::Dict{String, Int64})
    z_neighbor = copy(z)
    z_neighbor[move_dict["change_class"], move_dict["old_col"]] = 0
    z_neighbor[move_dict["change_class"], move_dict["new_col"]] = 1
    return z_neighbor
end


"""
Generates a starting feasible solution
"""
function gen_feasible_solution(rng::MersenneTwister, n_class::Int, n_label::Int)
    # Place n_label classes in a column so that we ensure we have a
    # feasible solution
    z = zeros(Int64, (n_class, n_label))
    classes = collect(1:n_class)
    initial_assignment = shuffle(rng, classes)[1:n_label]
    z[initial_assignment, 1:n_label] = 1
    filter!(x -> x ∉ initial_assignment, classes)

    # Place the remaining classes randomly
    random_assign = rand(rng, 1:n_label, length(classes))
    for (class, assignment) in zip(classes, random_assign)
        z[class, assignment] = 1
    end
    return z
end


"""
Infers the lone classes
"""
function infer_lone_class(z::Array{Int64, 2})
    lone_cols = find(sum(z, 1) .== 1)
    if length(lone_cols) == 0
        return Array{Int64, 1}(0)
    else
        lone_classes = map(col -> find(z[:, col] .== 1), lone_cols)
        return Iterators.flatten(lone_classes)
    end
end


"""
Infers the combo label map
"""
function infer_combos(z::Array{Int64, 2}, combo_cols::Array{Int64, 1})
    combos = map(col -> find(z[:, col] .> 0), combo_cols)
    combos = map(iter -> combinations(iter, 2), combos)
    return Iterators.flatten(combos)
end


"""
Infers the class and combo map in their entriety
"""
function infer_dvs(z::Array{Int64, 2})
    # Infer the lone class map
    lone_classes = infer_lone_class(z)

    # Infer the combinations that are present
    combo_cols = find(sum(z, 1) .> 1)
    if length(combo_cols) == 0
        combos = Array{Int64, 1}(0)
    else
        combos = infer_combos(z, combo_cols)
    end
    return lone_classes, combos
end


"""
Builds the array for the new columns in the neighborhood of z
"""
function build_new_cols(idx_arr::Array{Int64, 1}, n_label::Int)
    # new_cols = zeros(Int64, (length(j_idx) * (n_label - 1)))
    new_cols = Array{Int64, 1}(length(idx_arr) * (n_label - 1))
    j = 1
    for idx in idx_arr
        for i = 1:n_label
            if i == idx
                continue
            else
                new_cols[j] = i
                j += 1
            end
        end
    end
    return new_cols
end


"""
Computes the objective function from the entire similarity and maps
"""
function compute_objective(class_sim::Dict{Int64, Float32},
    combo_sim::Dict{Array{Int64, 1}, Float32},
    class_arr::Union{Iterators.Flatten, Array{Int64, 1}},
    combo_arr::Union{Iterators.Flatten, Array{Int64, 1}})

    lone_vals = map(idx -> class_sim[idx], class_arr)
    combo_vals = map(combo -> combo_sim[combo], combo_arr)
    return sum(lone_vals) + sum(combo_vals)
end


"""
Computes the change in objective value
"""
function compute_obj_change(old_combo_sim::Array{Float32, 1},
    new_combo_sim::Array{Float32, 1}, old_class_sim::Array{Float32, 1},
    new_class_sim::Array{Float32, 1})
    old_obj = sum(old_class_sim) + sum(old_combo_sim)
    new_obj = sum(new_class_sim) + sum(new_combo_sim)
    return new_obj - old_obj
end


"""
Finds the non-zero entries in a matrix
"""
function find_nonzero(z::Array{Int64, 2}, cols::Array{Int64, 1})
    n_class = size(z, 1)
    i_idx = Array{Int64, 1}(0)
    j_idx = Array{Int64, 1}(0)
    for col in cols
        for i = 1:n_class
            if z[i, col] == 1
                push!(i_idx, i)
                push!(j_idx, col)
            end
        end
    end
    return i_idx, j_idx
end


"""
Helper function repeat an array
"""
function repeat_array(arr::Array{Int64, 1}, repeats::Int)
    out = Array{Int64, 1}(length(arr) * repeats)
    j = 1
    for entry in arr
        for i = 1:repeats
            out[j] = entry
            j += 1
        end
    end
    return out
end


# Define an iterator so that we can build the move array without
# pay the price of transposing it and shuffling it
struct MoveArray
    z::Array{Int64, 2}
    rng::MersenneTwister
end


function Base.start(m::MoveArray)
    # Determine the columns with combinations and grab the corresponding
    # (i, j) indices
    n_label = size(m.z, 2)
    combo_cols = find(sum(m.z, 1) .> 1)
    i_idx, j_idx = find_nonzero(m.z, combo_cols)

    # Generate all possible columns we can move to
    new_cols = build_new_cols(j_idx, n_label)
    change_class = repeat_array(i_idx, n_label - 1)
    old_cols = repeat_array(j_idx, n_label - 1)

    # Define the random permutation which we will use to search
    # through the space and will help us define the end state
    # for the iterator
    search_order = shuffle(m.rng, 1:length(new_cols))
    state_dict = Dict([("new_cols", new_cols), ("change_class", change_class),
    ("old_cols", old_cols), ("search_order", search_order)])
    return state_dict
end


function Base.next(m::MoveArray, state::Dict)
    # Grab the index we will use to search for the given iteration
    idx = pop!(state["search_order"])
    move_dict = Dict([("change_class", state["change_class"][idx]),
    ("old_col", state["old_cols"][idx]), ("new_col", state["new_cols"][idx])])
    return (move_dict, state)
end


function Base.done(m::MoveArray, state::Dict)
    length(state["search_order"]) <= 0
end


"""
Build an array containing the combos that changed
"""
function build_change_combos(itr::Array{Int64, 1}, label::Int)
    change_combos = Array{Array{Int64, 1}, 1}(length(itr))
    for i = 1:length(itr)
        if label < itr[i]
            change_combos[i] = [label, itr[i]]
        else
            change_combos[i] = [itr[i], label]
        end
    end
    return change_combos
end


"""
Helper function to remove the label of interest from array of entries
"""
function remove_label(entries::Array{Int64, 1}, label::Int)
    out = Array{Int64, 1}(0)
    for entry in entries
        if entry != label
            push!(out, entry)
        end
    end
    return out
end


"""
Determine the combos that changed in the neighboring solution
"""
function find_combo_changes(z::Array{Int64, 2}, label::Int, col::Int)
    entries = find(z[:, col] .> 0)
    entries = remove_label(entries, label)
    return build_change_combos(entries, label)
end


"""
Determine the assignments that changed in the neighboring solution
"""
function find_assignment_changes(z::Array{Int64, 2},
    move_dict::Dict{String, Int64})
    # Find the old and new combos
    old_combos = find_combo_changes(z, move_dict["change_class"],
    move_dict["old_col"])

    new_combos = find_combo_changes(z, move_dict["change_class"],
    move_dict["new_col"])

    # Find the old and new lone classes
    orig_col_sum = sum(z, 1)
    new_col_sum = copy(orig_col_sum)
    new_col_sum[move_dict["new_col"]] += 1
    new_col_sum[move_dict["old_col"]] -= 1
    if (0 ∉ orig_col_sum) && (0 ∉ new_col_sum)
        old_class = Array{Int64, 1}(0)
        new_class = Array{Int64, 1}(0)
    else
        old_class = infer_lone_class(z)
        z_neighbor = gen_neighbor(z, move_dict)
        new_class = infer_lone_class(z_neighbor)
    end
    return [old_combos, new_combos, old_class, new_class]
end


"""
Helper function to quickly grab entries from dictionary
"""
function get_dict_values(dict::Dict, entries::Array{Any, 1})
    out = Array{Float32, 1}(length(entries))
    for i = 1:length(entries)
        out[i] = dict[entries[i]]
    end
    return out
end


"""
Gets the similarity values for the combinations and clases that changed
"""
function get_similarity(sim_dict::Dict, change_idx::Array)
    return get_dict_values(sim_dict, change_idx)
end


"""
Gets the similarity measures for all the assignment changes
"""
function get_change_similarity(combo_sim::Dict{Array{Int64, 1}, Float32},
    class_sim::Dict{Int64, Float32},
    change_arr::Array)
    sim_dicts = [combo_sim, combo_sim, class_sim, class_sim]
    similarity_values = map(get_similarity, sim_dicts, change_arr)
    dict_keys = ["old_combo_sim", "new_combo_sim", "old_class_sim",
    "new_class_sim"]
    return Dict(zip(dict_keys, similarity_values))
end


"""
Perfrom inexact search for a given starting point z
"""
function inexact_search(z::Array{Int64, 2}, rng::MersenneTwister,
    class_sim::Dict{Int64, Float32}, combo_sim::Dict{Array{Int64, 1}, Float32},
    max_iter::Int)

    change_z = true
    n_iter = 0
    z_best = copy(z)

    # Continuously loop until we either reach a local min or hit the
    # iteration limit
    while change_z
        change_z = false

        # Check if we've hit the upper iteration limit
        if n_iter >= max_iter
            println("Reached max iterations")
            break
        end

        # Define an iterator that will allow us to randomly search
        # through the neighborhood
        move_arr = MoveArray(z_best, rng)

        for move_dict in move_arr
            # Check if the proposed new solution improves the objective
            change_arr = find_assignment_changes(z_best, move_dict)
            sim_dict = get_change_similarity(combo_sim, class_sim, change_arr)
            obj_change = compute_obj_change(
            sim_dict["old_combo_sim"], sim_dict["new_combo_sim"],
            sim_dict["old_class_sim"], sim_dict["new_class_sim"]
            )

            if obj_change < 0
                z_best = gen_neighbor(z_best, move_dict)
                n_iter += 1
                change_z = true
                break
            end
        end
    end

    # Having reached the final solution we will compute the final objective
    # value
    class_map, combo_map = infer_dvs(z_best)
    obj_val = compute_objective(class_sim, combo_sim, class_map, combo_map)
    return Dict([("label_map", z_best), ("obj_val", obj_val),
    ("n_iter", n_iter)])
end


"""
Performs a single local search run
"""
function single_search(seed::Int, n_label::Int, n_class::Int,
    class_sim::Dict{Int64, Float32}, combo_sim::Dict{Array{Int64, 1}, Float32},
    max_iter::Int)

    # Generate the starting solution
    rng = MersenneTwister(seed)
    z = gen_feasible_solution(rng, n_class, n_label)

    # Perform the inexact, local search
    return inexact_search(z, rng, class_sim, combo_sim, max_iter)
end

function test(n_class::Int, class_sim::Dict{Int64, Float32},
    combo_sim::Dict{Array{Int64, 1}, Float32}, max_iter::Int)
    single_search(17, 14, n_class, class_sim, combo_sim, max_iter)
end


"""
Helper function to determine the best solution
"""
function find_best_soln(solns::Array)
    # Get the solution objective values
    obj_vals = map(x -> x["obj_val"], solns)
    best_soln = indmin(obj_vals)
    label_map = solns[best_soln]["label_map"]
    obj_val = solns[best_soln]["obj_val"]
    n_iter = solns[best_soln]["n_iter"]
    return Dict([("label_map", label_map), ("obj_val", obj_val),
    ("n_iter", n_iter)])
end


"""
Performs local search n_init times and finds the best solution
"""
function local_search(n_label::Int, class_sim::Dict{Int64, Float32},
    combo_sim::Dict{Array{Int64, 1}, Float32}; n_init::Int=10,
    max_iter::Int=100000)

    n_class = length(values(class_sim))

    # Perform local search n_init times
    args = Iterators.repeated.([n_label, n_class, class_sim, combo_sim,
    max_iter])
    solns = pmap(single_search, 1:n_init, args[1], args[2], args[3], args[4],
    args[5])

    # Determine the best solution
    return find_best_soln(solns)
end
end
