module RandomLabelMap

export create_label_map


function create_label_map(nclass::Int, nlabel::Int)
    # Define the empty label map
    z = zeros(Int64, (nclass, nlabel))

    # Generate a random permutation of assignments for each
    # label
    rng = MersenneTwister(17)
    random_assignment = rand(rng, 1:nlabel, nclass)

    # Update the values in the matrix
    for (i, assignment) in enumerate(random_assignment)
        z[i, assignment] = 1
    end
    return z
end

end
