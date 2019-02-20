#' Creates the Node Top 1 plot to compare the various hierarchical classifiers
#'
#' @param wd Working directory to find the data
#'
#' @return Nothing; saves the plot to disk
#' @importFrom rlang .data
#' @export
build_nt1 = function(wd) {
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

  # Filter so we're just working with node_top1 values
  df = dplyr::filter(df, .data$metric == 'node_top1')
  metric_names = c("node_top1" = "Node Top 1")
  estimator_names = c("log" = "LR", "rf" = "RF", 'knn' = 'KNN')

  # Create the NT1 plot
  p = ggplot2::ggplot(df, ggplot2::aes(x=.data$value, color=.data$method)) +
    ggplot2::geom_density(size=1) +
    ggplot2::facet_grid(estimator ~ metric, scales='free',
                        labeller=ggplot2::labeller(metric=metric_names,
                                                   estimator=estimator_names)) +
    ggplot2::labs(x='Value', y='Density',
                  title='Node Top 1 Comparison',
                  color='Hierarchical\nMethod') +
    ggplot2::scale_color_manual(values=c('FC' = '#e41a1c',
                                         'HC-KMC' = '#377eb8',
                                         'HC-CD' = '#4daf4a',
                                         'HC-LP' = '#984ea3',
                                         'HC-SC' = '#ff7f00')) +
    ggplot2::theme_bw()

  # Count the number of unique estimators in the data to scale the plot
  # appropriately
  n_estimators = length(unique(df$estimator))

  # Save the plot to disk
  base_folder = basename(wd)
  filename = paste(base_folder, 'nt1.pdf', sep='_')
  filepath = file.path(wd, 'figures', filename)
  R.devices::suppressGraphics(ggplot2::ggsave(filepath, plot=p, scale=0.7,
                                              height=(4 * n_estimators),
                                              width=12))
}
