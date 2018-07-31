module BenchmarkMethods

using Combinatorics

export benchmark


"""
Builds the array of true lone classes
"""
function find_lone_assignments(label_groups::AbstractArray)
    return [label for label in label_groups if length(label) == 1]
end


"""
Counts the number of correct combination assignments
"""
function count_correct(assign_group, true_groups)
    group_overlap = [intersect(assign_group, group) for group in true_groups]
    correct_arr = [binomial(length(iter), 2) for iter in group_overlap]
    return sum(correct_arr)
end


function benchmark(label_map::AbstractArray, label_groups::AbstractArray)

    # Determine which methods we need to depnding on what data was
    # passed to function
    nlabel = size(label_map, 2)
    groups = [find(label_map[:, i]) for i = 1:nlabel]

    # Find any lone class assignments
    println("Computing lone class accuracy")
    lone_assignments = find_lone_assignments(groups)
    true_lone = find_lone_assignments(label_groups)
    nlone = length(true_lone)
    if length(lone_assignments) == 0
        lone_correct = 0
    elseif nlone == 0
        lone_correct = 0
    else
        lone_correct = length(intersect(Iterators.flatten(lone_assignments),
        Iterators.flatten(true_lone)))
    end

    # Get the true assignment groups
    ncombo = sum([binomial(length(iter), 2) for iter in label_groups])

    # Get the correct number of assignments for both the label map and
    # kmeans
    println("Computing combination accuracy")
    combo_correct = sum(map(count_correct, groups,
    Iterators.repeated(label_groups)))

    # Finally determine the accuracy of each method
    accuracy = (combo_correct + lone_correct) / (ncombo + nlone)
    return accuracy
end

end
