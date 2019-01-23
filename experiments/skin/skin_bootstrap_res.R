# Get the package which will generate our results
library(getBootResults)

# Get command line arguments for the directory where data is saved
args = commandArgs(trailingOnly = T)

# Execute the source code and generate TeX tables
gen_all_tex_code(args[1])

# Generate the plots for each of the experiments
gen_boot_plots(args[1])
