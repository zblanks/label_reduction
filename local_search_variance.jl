module LocalSearch

using Distances
using StatsBase

export local_search


"""
Updates the min values in a vector
"""
function updatemin!(r::AbstractArray, x::AbstractArray)
    n = length(r)
    length(x) == n || throw(DimensionMismatch("Inconsistent array lengths."))
    @inbounds for i = 1:n
        xi = x[i]
        if xi < r[i]
            r[i] = xi
        end
    end
    return r
end


"""
Selects the L starting classes for the initial feasible solution
"""
function get_start_classes(X::AbstractArray, rng::MersenneTwister, nlabel::Int)
    # Select one of the classes at random to act as the starting point
    n = size(X, 2)
    start_classes = zeros(Int, (nlabel,))
    p = rand(rng, 1:n)
    start_classes[1] = p

    # Compute the columnwise distances from the starting point
    min_dists = colwise(SqEuclidean(), X, view(X, :, p))

    # Select the remaining classes according to the k-means++ rule
    tmp_dists = zeros(Float64, (n,))
    for i = 2:nlabel
        p = sample(rng, 1:n, weights(min_dists))
        start_classes[i] = p

        # Update the min distances
        colwise!(tmp_dists, SqEuclidean(), X, view(X, :, p))
        updatemin!(min_dists, tmp_dists)
    end
    return start_classes
end


"""
Generates a starting feasible solution
"""
function gen_feasible_solution(X::AbstractArray, rng::MersenneTwister,
    nclass::Int, nlabel::Int)
    # Place n_label classes in a column so that we ensure we have a
    # feasible solution according to the k-means++ rule
    z = zeros(Int64, (nclass, nlabel))
    classes = collect(1:nclass)
    initial_assignment = get_start_classes(X, rng, nlabel)
    for i = 1:nlabel
        z[initial_assignment[i], i] = 1
    end
    filter!(x -> x ∉ initial_assignment, classes)

    # Place the remaining classes proportional to their squared euclidean
    # distance
    col_dists = zeros(Float64, (nlabel,))
    for class in classes
        # Compute the colwise distances from the given class to the initial
        # assignment classes
        colwise!(col_dists, SqEuclidean(), view(X, :, initial_assignment),
        view(X, :, class))

        # Assign the label based off the closest centroid
        assignment = indmin(col_dists)
        z[class, assignment] = 1
    end
    return z
end


function compute_centroid(group::Vector{Int}, mean_dict::Dict,
    sample_dict::Dict)

    # Get the total number of samples for the group
    nsamples = sum([sample_dict[label] for label in group])

    # Compute the centroid vector
    μ = zeros(Float64, length(mean_dict[1]))
    @inbounds for label in group
        μ += (sample_dict[label] / nsamples) * mean_dict[label]
    end
    return Dict("centroid" => μ, "nsamples" => nsamples)
end


function sqeuclidean(x::AbstractArray, y::AbstractArray)
    dist = 0.0
    @simd for i = 1:length(x)
        @inbounds dist += (x[i] - y[i])^2
    end
    return dist
end


function compute_variance(group::Vector{Int}, centroid_dict::Dict,
    mean_dict::Dict, sample_dict::Dict)
    var = 0.0
    @simd for i = 1:length(group)
        @inbounds var += (sample_dict[group[i]] / centroid_dict["nsamples"]) *
        sqeuclidean(mean_dict[group[i]], centroid_dict["centroid"])
    end
    return var
end


function determine_changes(z::AbstractArray, move_dict::Dict,
    old_groups::AbstractArray)

    # Build the new groups
    new_groups = similar(old_groups)
    new_groups[1] = filter(x -> x ≠ move_dict["change_class"], old_groups[1])
    new_groups[2] = push!(copy(old_groups[2]), move_dict["change_class"])
    return new_groups
end


function check_var_change(old_groups::AbstractArray, new_groups::AbstractArray,
    old_centroids::AbstractArray, mean_dict::Dict, sample_dict::Dict)

    # Compute the relevant group centroids
    new_centroids = [compute_centroid(group, mean_dict, sample_dict)
    for group in new_groups]

    # Compute the old variance
    old_var = 0.0
    @inbounds for i = 1:length(old_centroids)
        old_var += compute_variance(old_groups[i], old_centroids[i],
        mean_dict, sample_dict)
    end

    # Compute the new variance
    new_var = 0.0
    @inbounds for i = 1:length(new_centroids)
        new_var += compute_variance(new_groups[i], new_centroids[i],
        mean_dict, sample_dict)
    end

    # Check if we improved
    return new_var - old_var
end


"""
Helper function to get non-zero indices of groups with more than one
entry to maintain feasibility
"""
function find_indices(z::AbstractArray, cols::AbstractArray)
    nclass = size(z, 1)
    i_idx = Array{Int, 1}(0)
    j_idx = Array{Int, 1}(0)
    for col in cols
        for i = 1:nclass
            if z[i, col] == 1
                push!(i_idx, i)
                push!(j_idx, col)
            end
        end
    end
    return i_idx, j_idx
end


"""
Builds the array for the new columns in the neighborhood of z
"""
function build_new_cols(idx_arr::Array{Int64, 1}, nlabel::Int)
    # new_cols = zeros(Int64, (length(j_idx) * (n_label - 1)))
    new_cols = Array{Int64, 1}(length(idx_arr) * (nlabel - 1))
    j = 1
    for idx in idx_arr
        for i = 1:nlabel
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


"""
Generates a neighbor from the current best solution
"""
function gen_neighbor(z::Array{Int64, 2}, move_dict::Dict{String, Int64})
    z_neighbor = copy(z)
    z_neighbor[move_dict["change_class"], move_dict["old_col"]] = 0
    z_neighbor[move_dict["change_class"], move_dict["new_col"]] = 1
    return z_neighbor
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
    nlabel = size(m.z, 2)
    combo_cols = find(sum(m.z, 1) .> 1)
    i_idx, j_idx = find_indices(m.z, combo_cols)

    # Generate all possible columns we can move to
    new_cols = build_new_cols(j_idx, nlabel)
    change_class = repeat_array(i_idx, nlabel - 1)
    old_cols = repeat_array(j_idx, nlabel - 1)

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


function Base.length(m::MoveArray)
    combo_cols = find(sum(m.z, 1) .> 1)
    combo_classes = [find(m.z[:, col]) for col in combo_cols]
    combo_classes = collect(Iterators.flatten(combo_classes))
    return length(combo_classes) * (size(m.z, 2) - 1)
end


"""
Performs inexact search given a starting point z
"""
function inexact_search(z::AbstractArray, rng::MersenneTwister,
    mean_dict::Dict, sample_dict::Dict, max_iter::Int)

    changez = true
    niter = 0
    zbest = copy(z)

    # Continuously loop until we reach a local min or hit the iteration
    # limit
    while changez
        changez = false

        # Check if we reached the iteration limit
        if niter >= max_iter
            println("Reached max iterations")
            break
        end

        # Define the iterator to search the space in parallel where
        # we will break on the first improvement
        move_arr = MoveArray(zbest, rng)

        # Compute the centroids for the current best feasible solution
        label_groups = [find(zbest[:, i]) for i = 1:size(zbest, 2)]
        centroids = [compute_centroid(group, mean_dict, sample_dict)
        for group in label_groups]

        for move_dict in move_arr
            old_groups = [label_groups[group] for group
            in [move_dict["old_col"], move_dict["new_col"]]]

            new_groups = determine_changes(zbest, move_dict, old_groups)
            old_centroids = [centroids[col] for col in [move_dict["old_col"],
            move_dict["new_col"]]]

            var_change = check_var_change(old_groups, new_groups, old_centroids,
            mean_dict, sample_dict)

            if var_change < 0
                if mod(niter, 100) == 0
                    println(string("Completed ", niter, " iterations"))
                end
                zbest = gen_neighbor(zbest, move_dict)
                niter += 1
                changez = true
                break
            end
        end
    end

    # We've reached a local min; find the final solution values
    final_groups = [find(zbest[:, col]) for col = 1:size(zbest, 2)]
    centroids = Vector{Dict}(length(final_groups))
    for i = 1:length(final_groups)
        centroids[i] = compute_centroid(final_groups[i], mean_dict, sample_dict)
    end

    var_values = Vector{Float64}(size(zbest, 2))
    for i = 1:size(zbest, 2)
        var_values[i] = compute_variance(final_groups[i], centroids[i],
        mean_dict, sample_dict)
    end
    return Dict("zbest" => zbest, "obj_val" => sum(var_values),
    "niter" => niter)
end


"""
Helper function to compute the mean vector for a given class
"""
function compute_mean_vector(X::AbstractArray)
    nrow = size(X, 1)
    return [mean(view(X, i, :)) for i = 1:nrow]
end


"""
Helper function to compute the mean vector and number of samples
for each provided class
"""
function build_req_dicts(X::AbstractArray)
    nrow = size(X, 1)
    nclass = length(unique(view(X, nrow, :)))

    # Compute the mean vector for all classes
    mean_vectors = Array{Array{Float64, 1}, 1}(nclass)
    idx_list = [find(view(X, nrow, :) .== i) for i = 1:nclass]
    sample_count = [length(idx) for idx in idx_list]
    @inbounds for (i, idx) in enumerate(idx_list)
        mean_vectors[i] = compute_mean_vector(view(X, 1:(nrow - 1), idx))
    end

    # Get the data into dictionaries for easy access
    mean_dict = Dict(zip(1:nclass, mean_vectors))
    sample_dict = Dict(zip(1:nclass, sample_count))
    return mean_dict, sample_dict
end


"""
Builds the matrix containing all the mean vectors
"""
function build_mean_arr(mean_dict::Dict)
    n = length(mean_dict[1])
    p = length(mean_dict)
    m = Array{AbstractFloat, 2}(n, p)
    for i = 1:p
        m[:, i] = mean_dict[i]
    end
    return m
end


"""
Performs local search ninit times and finds the best solution
"""
function local_search(X::AbstractArray, seed::Int, nlabel::Int;
    max_iter::Int=100000)

    # Build the required dictionaries needed for the search
    println("Determining mean vectors")
    mean_dict, sample_dict = build_req_dicts(X)

    nrow = size(X, 1)
    nclass = length(unique(view(X, nrow, :)))

    # Build the mean data array
    mean_arr = build_mean_arr(mean_dict)

    # Generate a starting solution
    println("Building starting feasible solution")
    rng = MersenneTwister(seed)
    z = gen_feasible_solution(mean_arr, rng, nclass, nlabel)

    # Perform inexact local search
    println("Beginning local search")
    return inexact_search(z, rng, mean_dict, sample_dict, max_iter)
end

end
