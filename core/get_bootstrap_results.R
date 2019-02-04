# Get the package which will generate our results
library(getBootResults)

# Get command line arguments for the directory where data is saved
args = commandArgs(trailingOnly = T)

# Generate the distributions for each of unique experiments
if (length(args) == 2) {
  gen_boot_plots(wd=args[1], max_combo_size=args[2])
} else {
  gen_boot_plots(wd=args[1])
}
