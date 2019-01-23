#' Converts the method names into a more read-able format for the plot title
#'
#' @param setting The particular setting for the given experiment
#'
#' @return More read-able string for the given method
convert_method_names = function(setting) {
  # Go through expected cases and adjust the name accordingly
  if (setting == "f") {
    return("Flat")
  } else if (setting == "hci") {
    return("Hierarchical Classifier")
  }
}

#' Creates and saves the bootstrap histograms to disk
#'
#' @param pair_id Unique experiment pair ID
#' @param boot_df Bootstrap results DataFrame
#' @param exp_diff_df Experiment difference DataFrame
#' @param savepath Directory in which to save the plot
#'
#' @return Nothing; saves the plot to disk
#'
#' @importFrom rlang .data
gen_boot_hists = function(pair_id, boot_df, exp_diff_df, savepath) {
  # Filter the experiment differences based on the particular pair ID
  new_exp_diff_df = dplyr::filter(exp_diff_df, .data$pair_id == !!pair_id)

  # Using the experiment difference we need to infer the correct title
  # and caption for our plot so that they're able to convey the correct
  # information
  if (new_exp_diff_df$distinguisher == "method") {
    # Get the first and second different classifier types
    settings = c(new_exp_diff_df$setting1, new_exp_diff_df$setting2)
    method_names = purrr::map_chr(.x=settings, .f=convert_method_names)

    # Generate the correct plot title and caption based off the method names
    plot_title = paste(method_names[2], "vs.", method_names[1],
                       "Boostrap Distributions")
    caption_title = paste("Percent difference calculated with respect to",
                          method_names[2])
  }

  # Generate a list that will allow us to re-name the facets to explain
  # them more clearly
  facet_names = c("leaf_top1" = "Leaf Top 1", "leaf_top3" = "Leaf Top 3",
                  "node_top1" = "Node Top 1")

  # Now we can generate the plot using the inferred plot and caption titles
  new_df = dplyr::filter(boot_df, .data$pair_id == !!pair_id)
  p = ggplot2::ggplot(new_df, ggplot2::aes(x=.data$value, group=.data$metric)) +
    ggplot2::geom_histogram(color="white") +
    ggplot2::facet_wrap(~metric, scales="free",
                        labeller=ggplot2::labeller(metric=facet_names)) +
    ggplot2::labs(x="Percent Difference", y="Count", title=plot_title,
                  caption=caption_title) +
    ggplot2::theme_bw()

  # Save the plot to disk
  filepath = file.path(savepath, "figures", paste(pair_id, ".pdf", sep=""))
  ggplot2::ggsave(filepath, scale=0.8, height=5, width=12)
}

#' Generates all of the bootstrap histograms
#'
#' @param wd Directory containing the files needed to generate the plots
#'
#' @return Nothing; saves the plots to disk
#' @export
gen_boot_plots = function(wd) {
  # Get the bootstrap results as well as the files needed to get the
  # unique experiments
  boot_df = readr::read_csv(file.path(wd, "boot_res.csv"))
  pair_df = readr::read_csv(file.path(wd, "exp_pairs.csv"))
  exp_df = readr::read_csv(file.path(wd, "experiment_settings.csv"))

  # Get the experiment differences
  uniq_ids = get_uniq_ids(boot_df)
  exp_diff_df = purrr::map_df(.x=uniq_ids,
                              .f=~get_exp_diff(.x, pair_df, exp_df))

  # Go through each unique pair_id in the exp_diff and generate and save
  # the plot to disk
  dir.create(file.path(wd, "figures"), showWarnings=F)
  purrr::map(.x=uniq_ids, .f=~gen_boot_hists(.x, boot_df, exp_diff_df, wd))
}
