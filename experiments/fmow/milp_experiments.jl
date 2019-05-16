using HDF5, ArgParse, DataFrames, CSV
include("/home/zblanks/label_reduction/core/label_group_milp.jl")

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--nlabels"
            help="Number of meta-classes to infer with the MILP"
            arg_type=Int
        "--file_loc"
            help="Location of the MILP functions"
            arg_type=String
        "--wd"
            help="Location to save the results"
            arg_type=String
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()

    # # Get the label_group_milp functions
    # include(args["file_loc"])

    # Get the data
    println("Getting data")
    data = h5open(joinpath(args["wd"], "train.h5"), "r") do file
        read(file)
    end
    X = transpose(data["X_train"])
    y = vec(data["y_train"])

    # Build the path for the log file from the MILP
    log_file = joinpath(args["wd"], "milp_res", string(args["nlabels"], ".txt"))

    # Run the MILP label grouping program
    println("Running MILP")
    start_time = time()
    z, obj_val = group_labels(X, y, args["nlabels"], log_file)
    stop_time = time() - start_time

    # Combine the results in a DataFrame for easier analysis
    df = DataFrame(k=args["nlabels"], search_time=stop_time, obj_val=obj_val)

    # Save the results to disk
    savepath = joinpath(args["wd"], "milp_res", "milp_res.csv")

    # If the file already exists, then we will append to it; otherwise
    # we need to start the new file
    if isfile(savepath)
        CSV.write(df, append=true)
    else
        CSV.write(df)
    end
end

main()
