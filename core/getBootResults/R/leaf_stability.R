#' Generates the leaf entropy plot
#'
#' @param wd Working directory to find the data
#' @param ... Additional arguments that can be passed for filtering
#'
#' @return Nothing; saves the plot to disk
#' @importFrom rlang .data
#' @export
gen_leaf_plot = function(wd, ...) {
  # Get the data
  exp_df = readr::read_csv(file.path(wd, 'experiment_settings.csv'),
                           col_types=readr::cols())

  leaf_df = readr::read_csv(file.path(wd, 'leaf_stability_res.csv'),
                            col_types=readr::cols())

  # Combine the data to get the experimental settings
  df = dplyr::inner_join(exp_df, leaf_df, by='id')

  # Adjust the method in the DataFrame so it's easier to compare the
  # approaches
  df = dplyr::mutate(df, method = purrr::map2_chr(.data$method,
                                                  .data$group_algo,
                                                  adjust_method))

  # Perform any additional filtering operations
  addl_args = list(...)
  df = filter_df(df, addl_args)

  # Generate a summary value for each method and estimator
  df = tidyr::spread(df, key=.data$metric, value=.data$value)
  df = dplyr::filter(df, .data$label == -1)
  df = dplyr::group_by(df, .data$method, .data$estimator)
  df = dplyr::summarize(df, entropy = stats::median(.data$med_entropy),
                        logloss = stats::median(.data$log_loss))

  estimator_names = c("log" = "LR", "rf" = "RF", 'knn' = 'KNN')

  # Make the plot
  p = ggplot2::ggplot(df, ggplot2::aes(x=.data$entropy, y=.data$logloss,
                                       color=.data$method)) +
    ggplot2::geom_point(size=6) +
    ggplot2::facet_grid(estimator ~ ., scales='free',
                        labeller=ggplot2::labeller(estimator=estimator_names)) +
    ggplot2::labs(x='Median Posterior Entropy', y='Median Log-Loss',
                  title='Posterior Distribution Comparison',
                  color='Training\nMethod') +
    ggplot2::scale_color_manual(values=c('FC' = '#e41a1c',
                                         'KMC' = '#377eb8',
                                         'CD' = '#4daf4a',
                                         'LP' = '#984ea3',
                                         'SC' = '#ff7f00')) +
    ggplot2::theme_bw()

  # Save the plot to disk
  savepath = get_plot_name(addl_args)
  base_folder = basename(wd)
  filename = paste(base_folder, savepath, sep='-')
  filepath = file.path(wd, 'figures', filename)
  R.devices::suppressGraphics(ggplot2::ggsave(filepath, plot=p, scale=0.5,
                                              height=10, width=16))
}
