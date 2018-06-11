using JuMP
using Gurobi
using Combinatorics


function build_combo_set(C::Int, L::Int)
    """Defines the combination set needed for our MIP

    Args:
        C = the original number of classes
        L = the desired number of new classes

    Returns:
        (Array): {choose(C-L+1-i, 2) : 0 <= i <= C-L, i ∈ Ζ}
    """

    # Define the appropriate range of i values
    i_vals = 0:(C-L-1)
    combo_set = Vector(length(i_vals))
    for (i, val) in enumerate(i_vals)
        combo_set[i] = binomial(C-L+1-val, 2)
    end
    return combo_set
end



function label_reduce(C::Int, L::Int, t::Array, r::Array, λ::Array, path::String)
    """Generates and runs the label reduction IP

    Args:
        C = the original number of classes
        L = the desired number of new classes
        t = the similiarity measure between class combinations
        r = the original class similiarity measure
        λ = correction factor for pairwise approximation
        path = location to save the solver log file

    Returns:
        (Array): z_{i,j} for all i, j
    """

    # We need to determine the total number of class combinations
    n_comb = binomial(C, 2)

    # We also need to compute the set defining the number of pairwise
    # combination options
    combo_set = build_combo_set(C, L)

    # Check that the provided arrays have the correct dimension
    @assert size(t) == (n_comb, L)
    @assert size(r) == (C, L)
    @assert size(k) == (length(combo_set), 1)

    # Define the model
    m = Model(solver=GurobiSolver(LogFile=path, TimeLimit=5*60))

    # Define the variables
    @variable(m, z[i=1:C, j=1:L], Bin)
    @variable(m, x[j=1:L], Bin)
    @variable(m, 0 <= y[i=1:C, j=1:L] <= 1)
    @variable(m, w[s=1:n_comb, j=1:L], Bin)
    @variable(m, 0 <= γ[s=1:n_comb, j=1:L, k=combo_set] <= 1)
    @variable(m, θ[j=1:L, k=combo_set], Bin)
    @variable(m, 0 <= δ[j=1:L, k=combo_set] <= 1)

    # Assignment constraints
    @constraint(m, [i=1:C], sum(z[i, j=1:L]) == 1)
    @constraint(m, [j=1:L], sum(z[i=1:C, j]) >= 1)

    # Logic for determining when a column is by itself
    M = C - L + 1
    @constraint(m, [j=1:L], sum(z[i=1:C, j]) <= 1 + M*(1-x[j]))
    @constraint(m, [j=1:L], sum(z[i=1:C, j]) >= 2 - M*x[j])

    # Linearize the lone class similiarty variable
    @constraint(m, [i=1:C, j=1:L], y[i, j] <= z[i, j])
    @constraint(m, [i=1:C, j=1:L], y[i, j] <= x[j])
    @constraint(m, [i=1:C, j=1:L], y[i, j] >= z[i, j] + x[j] - 1)

    # Define the auxiliary variables for the presence of combinations
    a = collect(combinations(1:C, 2))
    for (i, elt) in enumerate(a)
        append!(elt, i)
    end
    for j in 1:L
        for elt in a
            @constraint(m, z[elt[1], j] + z[elt[2], j] >= 2 - 2*(1 - w[elt[3], j]))
            @constraint(m, z[elt[1], j] + z[elt[2], j] <= 1 + 2*w[elt[3], j])
        end
    end

    # Linearize the variable for the combination similiarity
    @constraint(m, [s=1:n_comb, j=1:L, k=combo_set], γ[s, j, k] <= θ[j, k])
    @constraint(m, [s=1:n_comb, j=1:L, k=combo_set], γ[s, j, k] <= w[s, j])
    @constraint(m, [s=1:n_comb, j=1:L, k=combo_set], γ[s, j, k] >= θ[j, k] + w[s, j] - 1)

    # Define the constraint which ensures that we are getting the average
    @constraint(m, [j=1:L], sum(sum(γ[s, j, k]/k for k in combo_set) for s=1:n_comb) <= 1)
    @constraint(m, [j=1:L], sum(θ[j, k] for k in combo_set) == 1)

    # Add the correction constraints
    @constraint(m, [j=1:L, k=combo_set], δ[j, k] <= θ[j, k])
    @constraint(m, [j=1:L, k=combo_set], δ[j, k] <= x[j])
    @constraint(m, [j=1:L, k=combo_set], δ[j, k] >= θ[j, k] + x[j] - 1)

    # Define the Objective
    @objective(m, Max, vecdot(r, y) +
    sum(sum(sum((t[s, j]*γ[s, j, k])/k for k in combo_set) for s=1:n_comb) for j=1:L)) -
    sum(sum(λ[k]*(θ[j, k] - δ[j, k]) for k in combo_set) for j=1:L)

    # Solve the problem
    solve(m)
    return getvalue(z)
end
