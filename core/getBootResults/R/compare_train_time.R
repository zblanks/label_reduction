#' Updates the methods for a DataFrame
#'
#' @param df DataFrame to update the method column
#' @param exp_df Experiment settings DataFrame
#'
#' @return DataFrame with updated methods
#' @importFrom rlang .data
#' @keywords internal
update_method = function(df, exp_df) {
  # First combine the DataFrames
  df = dplyr::inner_join(df, exp_df, by='id')

  # Update the methods according to the interal adjustment method
  df = dplyr::mutate(df, method = purrr::map2_chr(.data$method,
                                                  .data$group_algo,
                                                  adjust_method))
  return(df)
}

#' Helper function to compute the total training time for a the HC DataFrame
#'
#' @param df HC time results DataFrame
#'
#' @return DataFrame containing only the total training time for a given
#' experiment
#'
#' @importFrom rlang .data
#' @keywords internal
compute_total_time = function(df) {
  # In this instance I only care about the timing metrics; not the other
  # performance metrics
  df = dplyr::filter(df, .data$metric %in% c('cluster_time', 'train_time'))

  # For the HC train time DataFrames, we need to spread the data and then
  # generate a new column detailing the total training time
  df = tidyr::spread(df, .data$metric, .data$value)
  df = dplyr::group_by(df, .data$run_num, .data$method, .data$estimator)
  df = dplyr::summarize(df, total_time = sum(.data$train_time +
                                               .data$cluster_time))
  df = tidyr::gather(df, key='metric', value='value', .data$total_time)
  return(df)
}

#' Helper function to get the final experiment IDs for the HC DataFrame
#'
#' @param df HC DataFrame
#'
#' @return DataFrame containing the final experiment IDs
#' @importFrom rlang .data
#' @keywords internal
get_final_ids = function(df) {
  # By construction, the final ID will correspond to the entry which has the
  # largest leaf_top1 for a given experiment (which is defined by the
  # run_num, method, and estimator)
  df = tidyr::spread(df, .data$metric, .data$value)
  df = dplyr::group_by(df, .data$run_num, .data$method, .data$estimator)
  df = dplyr::filter(df, .data$top1 == max(.data$top1))
  df = dplyr::ungroup(df)
  return(dplyr::select(df, .data$id, .data$run_num, .data$method,
                       .data$estimator))
}

#' Helper function which combines the metric.x/y and value.x/y in the
#' DataFrame
#'
#' @param df DataFrame containing the multiple metric and value columns
#'
#' @return DataFrame with a single metric and value column
#' @importFrom rlang .data
#' @keywords internal
combine_cols = function(df) {
  # Grab and combine the metric and value columns
  metric_col = c(rbind(df$metric.x, df$metric.y))
  value_col = c(rbind(df$value.x, df$value.y))

  # Drop the metric and value columns from the DataFrame
  df = dplyr::select(df, -c(.data$metric.x, .data$metric.y, .data$value.x,
                            .data$value.y))

  # Grab the IDs
  exp_ids = dplyr::pull(df, .data$id)

  # Since we are effectively doubling the metric and value columns we need to
  # repeat each ID twice
  exp_ids = rep(exp_ids, each=2)
  combine_df = dplyr::tibble(id=exp_ids, metric=metric_col, value=value_col)

  # Join the combine_df with df to have the final DataFrame
  df = dplyr::inner_join(df, combine_df, by='id')
  return(df)
}

#' Helper function to make the time comparison plot
#'
#' @param df DataFrame containing the data needed to generate the plot
#' @param wd Location to save the plot
#' @param metric Which metric to use for the time comparison plot
#'
#' @return ggplot2 object
#' @importFrom rlang .data
#' @keywords internal
make_plot = function(df, wd, metric) {
  # Define the abbreviations for the various estimators
  estimator_names = c("log" = "LR", "rf" = "RF", 'knn' = 'KNN')

  # Adjust the metric name for the title labels based on what is passed to
  # the function
  if (metric == 'leaf_top1') {
    metric_name = "Leaf Top 1"
  } else if (metric == 'leaf_top3') {
    metric_name = "Leaf Top 3"
  }

  # Now we can generate the series of plots comparing the experiments
  df = dplyr::filter(df, .data$estimator != 'log')
  p = ggplot2::ggplot(df, ggplot2::aes(x=log(.data$total_time), y=get(!!metric),
                                       color=.data$method)) +
    ggplot2::geom_point() +
    ggplot2::facet_grid(estimator ~ ., scales='free',
                        labeller=ggplot2::labeller(estimator=estimator_names)) +
    ggplot2::labs(x='Log(Total Training Time)', y=paste(metric_name, 'Value'),
                  title='Training Time Comparison',
                  color='Training\nMethod') +
    ggplot2::scale_color_manual(values=c('FC' = '#e41a1c',
                                         'HC-KMC' = '#377eb8',
                                         'HC-CD' = '#4daf4a',
                                         'HC-LP' = '#984ea3',
                                         'HC-SC' = '#ff7f00')) +
    ggplot2::theme_bw()

  # Save the plot disk
  n_estimators = length(unique(df$estimator))
  filepath = file.path(wd, 'figures', 'time_compare.pdf')
  R.devices::suppressGraphics(ggplot2::ggsave(filepath, plot=p, scale=0.7,
                                              height=(4 * n_estimators),
                                              width=12))
}

#' Generates the time comparison plot among the various methods
#'
#' @param wd Working directory to find the data
#' @param metric Which metric to use for the time comparison plot
#'
#' @return Nothing; saves the plot to disk
#' @importFrom rlang .data
#' @export
gen_time_plot = function(wd, metric='leaf_top1') {
  # To generate this plot we will need the experiment settings as well as
  # the timing results for the flat and HCs
  exp_df = readr::read_csv(file.path(wd, 'experiment_settings.csv'))
  fc_df = readr::read_csv(file.path(wd, 'fc_prelim_res.csv'))
  search_df = readr::read_csv(file.path(wd, 'search_res.csv'))
  raw_df = readr::read_csv(file.path(wd, 'raw_res.csv'))

  # Next we need to add the experimental settings to both of the DataFrames
  # (we can't join them just yet because fc_df does not have cluster_time)
  # and so this will lead to issues later if we join them now
  search_df = update_method(search_df, exp_df)
  fc_df = update_method(fc_df, exp_df)

  # Get the final experiment IDs for the search_df so that we can add the
  # out-of-sample performance metrics
  exp_ids = get_final_ids(search_df)

  # Compute the total training time for the HC and change the metric train_time
  # total time for the FC because they are equivalent and we will need it for
  # plotting
  search_df = compute_total_time(search_df)
  fc_df = dplyr::mutate(fc_df, metric = ifelse(.data$metric == 'train_time',
                                               'total_time', .data$metric))

  # Add the IDs from the exp_ids DataFrame
  search_df = dplyr::inner_join(search_df, exp_ids, by=c('run_num', 'method',
                                                         'estimator'))

  # Get the intersection of the columns for the two DataFrames so that we
  # joining the same columns
  like_cols = intersect(colnames(search_df), colnames(fc_df))
  search_df = dplyr::select(search_df, !!like_cols)
  fc_df = dplyr::select(fc_df, !!like_cols)
  df = dplyr::bind_rows(fc_df, search_df)

  # In addition we need to add the performance metrics to our DataFrame
  # to generate the time plot
  raw_df = dplyr::filter(raw_df, .data$metric == !!metric)
  df = dplyr::inner_join(df, raw_df, by='id')
  df = combine_cols(df)

  # Finally we can generate the time comparison plot
  df = tidyr::spread(df, .data$metric, .data$value)
  make_plot(df, wd, metric)
}
