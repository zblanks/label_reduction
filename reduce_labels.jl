using JuMP
using Gurobi
using Combinatorics


function label_reduce(C::Int, L::Int, t::Array, r::Array, f::Float64,
    path::String)
    """Generates and runs the label reduction IP

    Args:
        C = the original number of classes
        L = the desired number of new classes
        t = the similiarity measure between class combinations
        r = the original class similiarity measure
        f = the mixing factor
        path = location to save the solver log file

    Returns:
        Dict:
            Dictionary containing the optimal label map and the objective
            value of the solution
    """

    # We need to determine the total number of class combinations
    n_class_combo = binomial(C, 2)
    n_label_combo = binomial(L, 2)
    M = C - L + 1
    W = (M - 1) * (L - 1)

    # Check that the provided arrays have the correct dimension
    @assert size(t) == (n_class_combo, L)
    @assert size(r) == (C, L)

    # Define the model
    m = Model(solver=GurobiSolver(LogFile=path, TimeLimit=10*60,
    LogToConsole=0))

    # Define the variables
    @variable(m, z[i=1:C, j=1:L], Bin)
    @variable(m, x[j=1:L], Bin)
    @variable(m, w[s=1:n_class_combo, j=1:L], Bin)
    @variable(m, 0 <= y[i=1:C, j=1:L] <= 1)
    @variable(m, 0 <= τ[k=1:n_label_combo] <= M - 1)

    # Assignment constraints
    @constraint(m, [i=1:C], sum(z[i, j=1:L]) == 1)
    @constraint(m, [j=1:L], sum(z[i=1:C, j]) >= 1)

    # Label balance constraints by first building the K label combination
    # set and then adding the constraints
    label_combos = collect(combinations(1:L, 2))
    k_set = Array{Tuple}(length(label_combos),)
    for (i, elt) in enumerate(label_combos)
        k_set[i] = (elt[1], elt[2], i)
    end
    for elt in k_set
        @constraint(m, sum(z[i=1:C, elt[1]]) - sum(z[i=1:C, elt[2]])
        <= τ[elt[3]])
        @constraint(m, sum(z[i=1:C, elt[2]]) - sum(z[i=1:C, elt[1]])
        <= τ[elt[3]])
    end
    @constraint(m, sum(τ[1:n_label_combo]) <= f*W)


    # Logic for determining when a column is by itself
    @constraint(m, [j=1:L], sum(z[i=1:C, j]) <= 1 + M*(1-x[j]))
    @constraint(m, [j=1:L], sum(z[i=1:C, j]) >= 2 - M*x[j])

    # Linearize the lone class similiarty variable
    @constraint(m, [i=1:C, j=1:L], y[i, j] <= z[i, j])
    @constraint(m, [i=1:C, j=1:L], y[i, j] <= x[j])
    @constraint(m, [i=1:C, j=1:L], y[i, j] >= z[i, j] + x[j] - 1)

    # Define the auxiliary variables for the presence of combinations by
    # first building the set of class combinations and then adding the
    # constraints
    class_combos = collect(combinations(1:C, 2))
    s_set = Array{Tuple}(length(class_combos),)
    for (i, elt) in enumerate(class_combos)
        s_set[i] = (elt[1], elt[2], i)
    end
    for j = 1:L
        for elt in s_set
            @constraint(m, z[elt[1], j] + z[elt[2], j] >= 2 - 2(1-w[elt[3], j]))
            @constraint(m, z[elt[1], j] + z[elt[2], j] <= 1 + 2*w[elt[3], j])
        end
    end

    # Define the objective function
    @objective(m, Max, vecdot(r, y) + vecdot(t, w))

    # Solve the problem
    solve(m)
    soln_dict = Dict("map" => getvalue(z), "obj_val" => getobjectivevalue(m))
    return soln_dict
end
