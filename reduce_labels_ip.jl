using JuMP
using Gurobi
using Combinatorics


function label_reduce(C::Int, k::Int, t::Array, r::Array, path::String)
    """Generates and runs the label reduction IP

    Args:
        C = the original number of classes
        k = the desired number of new classes
        t = the similiarity measure between class combinations
        r = the original class similiarity measure

    Returns:
        (Array): z_{p,l} for all p, l
    """

    # We need to determine the total number of class combinations
    n_comb = binomial(C, 2)

    # Check that the provided arrays have the correct dimension
    @assert size(t) == (n_comb, k)
    @assert size(r) == (C, 1)

    # Define the model
    m = Model(solver=GurobiSolver(OutputFlag=0, LogFile=path))

    # Define our variables
    @variable(m, z[p=1:C, l=1:k], Bin)
    @variable(m, x[l=1:k], Bin)
    @variable(m, δ[p=1:C, l=1:k], Bin)
    @variable(m, y[p=1:C], Bin)
    @variable(m, w[s=1:n_comb, l=1:k], Bin)

    # Assign to only class and each label must have at least one class
    @constraint(m, [p=1:C], sum(z[p, l=1:k]) == 1)
    @constraint(m, [l=1:k], sum(z[p=1:C, l]) >= 1)

    # Logic for determining when a label has only one class in it
    @constraint(m, [l=1:k], sum(z[p=1:C, l]) >= 2 - x[l])
    M = (C - k + 1) - 2
    @constraint(m, [l=1:k], sum(z[p=1:C, l]) - 2 <= M*(1-x[l]) - x[l])
    @constraint(m, [l=1:k, p=1:C], z[p, l] + x[l] >= 2δ[p, l])
    @constraint(m, [p=1:C], y[p] == sum(δ[p, l=1:k]))

    # Logical constraints for the siutation where we have more than
    # one class in a label
    a = collect(combinations(1:C, 2))
    for (i, elt) in enumerate(a)
        append!(elt, i)
    end
    for l in 1:k
        for elt in a
            @constraint(m, z[elt[1], l] + z[elt[2], l] >= 2w[elt[3], l])
        end
    end

    # Objective
    @objective(m, Max, vecdot(t, w) + vecdot(r, y))

    # Solve the problem
    solve(m, relaxation=true)
    return getvalue(z)
end
