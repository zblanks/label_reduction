#' Helper function to adjust the method in the DataFrame to help us
#' distinguish when plotting
#'
#' @param method The method for the particular experiment
#' @param group_algo The grouping algorithm used for the experiment
#'
#' @return Adjusted DataFrame w/ new methods
#' @keywords internal
adjust_method = function(method, group_algo) {
  # Establish of series of if-then statements to generate the new method
  if (method == 'f') {
    return('FC')
  } else {
    # Adjust depending on the grouping algorithm
    if (group_algo == 'kmm') {
      return('HC-KMC')
    } else if (group_algo == 'comm') {
      return('HC-CD')
    } else if (group_algo == 'lp') {
      return('HC-LP')
    } else {
      return('HC-SC')
    }
  }
}

#' Helper function to do additional filtering of a DataFrame based on ...
#' argments
#'
#' @param df The DataFrame to filter
#' @param addl_args Additional arguments passed to the plot function call
#'
#' @return Updated DataFrame
#' @importFrom rlang .data
filter_df = function(df, addl_args) {
  if (!is.null(addl_args[['filter_args']])) {
    filter_args = addl_args$filter_args

    # Update the DataFrame
    for (i in 1:length(filter_args)) {
      col_name = filter_args[[i]]$col_name
      val = filter_args[[i]]$val
      df = dplyr::filter(df, (!!rlang::sym(col_name)) != val)
    }
  }

  return(df)
}

#' Helper function to get the plot name from the ... argument
#'
#' @param addl_args Additional arguments passed by ...
#'
#' @return Plot name
get_plot_name = function(addl_args) {
  if (!is.null(addl_args[['save_path']])) {
    save_path = addl_args$save_path
  } else {
    save_path = 'boot_distn.pdf'
  }
  return(save_path)
}

#' Produces the bootstrap distribution plot for a series of experiments
#'
#' @param wd Working directory to find the data
#' @param ... Additional arguments that can be passed for filtering
#'
#' @return Nothing; saves the plot to disk
#' @importFrom rlang .data
#' @export
gen_boot_distn = function(wd, ...) {
  # Get the bootstrap results and DataFrame mapping the experiment settings
  boot_df = readr::read_csv(file.path(wd, 'boot_res.csv'),
                            col_types=readr::cols())
  exp_df = readr::read_csv(file.path(wd, 'experiment_settings.csv'),
                           col_types=readr::cols())

  # Join the DataFrames so that we get the experimental settings
  df = dplyr::inner_join(boot_df, exp_df, by=c('exp_id' = 'id'))

  # Adjust the method in the in the DataFrame to make it easier to plot
  df = dplyr::mutate(df, method = purrr::map2_chr(.data$method,
                                                  .data$group_algo,
                                                  adjust_method))

  # Define the labeller for the estimator and the metrics
  metric_names = c("leaf_top1" = "Leaf Top 1", "leaf_top3" = "Leaf Top 3")
  estimator_names = c("log" = "LR", "rf" = "RF", 'knn' = 'KNN')

  # Now we can generate the series of plots comparing the experiments
  df = dplyr::filter(df, .data$metric != 'node_top1')

  # Check for any additional filtering that needs to be done via the ...
  # argument; we're assuming that a list is passed that provides the
  # col_name and the value that we want to filter with !=
  addl_args = list(...)
  df = filter_df(df, addl_args)

  p = ggplot2::ggplot(df, ggplot2::aes(x=.data$value, color=.data$method)) +
    ggplot2::geom_density(size=1) +
    ggplot2::facet_grid(estimator ~ metric, scales='free',
                        labeller=ggplot2::labeller(metric=metric_names,
                                                   estimator=estimator_names)) +
    ggplot2::labs(x='Value', y='Density',
                  title='Bootstrap Distribution Comparison',
                  color='Training\nMethod') +
    ggplot2::scale_color_manual(values=c('FC' = '#e41a1c',
                                         'HC-KMC' = '#377eb8',
                                         'HC-CD' = '#4daf4a',
                                         'HC-LP' = '#984ea3',
                                         'HC-SC' = '#ff7f00')) +
    ggplot2::theme_bw()

  # Check if a different name has been provided otherwise use the default
  # "boot_distn.pdf"
  save_path = get_plot_name(addl_args)

  # Save the plot to disk
  base_folder = basename(wd)
  filename = paste(base_folder, save_path, sep='-')
  filepath = file.path(wd, 'figures', filename)
  R.devices::suppressGraphics(ggplot2::ggsave(filepath, plot=p, scale=0.5,
                                              height=10, width=16))
}
