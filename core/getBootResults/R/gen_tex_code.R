#' Computes the average difference for a given metric (ex: leaf_top1)
#'
#' @param boot_df Bootstrap results DataFrame
#' @param id Unique experiment pair ID
#' @param metric Metric for the particular experiment (ex: leaf_top1)
#'
#' @return DataFrame containing the ID the metric, and the mean difference
#'
#' @importFrom rlang .data
#' @keywords internal
get_avg_diff = function(boot_df, id, metric) {

  # Get the vector of bootstrap values for the (ID, metric) combo
  new_df = dplyr::filter(boot_df, ((.data$pair_id == !!id) &
                                     (.data$metric == !!metric)))
  val_vect = dplyr::pull(new_df, .data$value)

  # Get the mean vector for the given (ID, metric) combination
  mean_val = mean(val_vect)

  # Generate a new results DataFrame containing the ID and the values for the
  # metric
  return(dplyr::tibble(id=id, metric=metric, value=mean_val))
}

#' Gets the unique experiment IDs
#'
#' @param df DataFrame containing a column with the experiment pair IDs
#'
#' @return Vector containing the unique pair IDs
#' @importFrom rlang .data
#' @keywords internal
get_uniq_ids = function(df) {
  uniq_ids = dplyr::distinct(df, .data$pair_id)
  return(dplyr::pull(uniq_ids))
}

#' Gets the bootstrap results for all the metrics
#'
#' @param boot_df Bootstrap results DataFrame
#'
#' @return A wide DataFrame containing the average difference for all
#' experiments
#'
#' @importFrom rlang .data
#' @keywords internal
get_boot_table = function(boot_df) {

  # First infer all unique (ID, metric) combinations
  uniq_ids = get_uniq_ids(boot_df)

  uniq_metrics = dplyr::distinct(boot_df, .data$metric)
  uniq_metrics = dplyr::pull(uniq_metrics)

  all_combos = expand.grid(id=uniq_ids, metric=uniq_metrics)

  # Map over all combinations and then combine the final DataFrame
  res_df = purrr::map2_df(.x=all_combos$id, .y=all_combos$metric,
                          .f=~get_avg_diff(boot_df, .x, .y))

  # res_df is in a long format and we need to convert it to a wide format
  # to be easier to generate a LaTeX table
  new_df = tidyr::spread(res_df, .data$metric, .data$value)
  return(new_df)
}

#' Generate the TeX code for the bootstrap results
#'
#' @param boot_df Bootstrap results DataFrame
#'
#' @return LaTeX code which can be saved to disk
#' @keywords internal
gen_boot_res_tex = function(boot_df) {
  # First we need to get the results DataFrame
  df = get_boot_table(boot_df)

  # We need to convert the ID from its long string to a {1, 2, ...} so that
  # it's more read-able
  df = dplyr::mutate(df, id = 1:nrow(df))

  # Using this table, we will now generate LaTeX code which will create a
  # nice table
  tex_code = knitr::kable(df, format="latex", caption="Bootstrap Results",
                          digits=3, booktabs=T,
                          col.names=c("Experiment #", "LT1", "LT3", "NT1"))

  return(tex_code)
}

#' Infers the unique experiments that we will use to generate the TeX code
#' and the plots
#'
#' @param pair_id Unique experiment pair ID
#' @param pair_df DataFrame defining the unique experiment pairs
#' @param exp_df DataFrame detailing all experiment settings
#'
#' @return DataFrame containing the unique experiments and what distinguishes
#' them
#'
#' @importFrom rlang .data
#' @keywords internal
get_exp_diff = function(pair_id, pair_df, exp_df) {
  # Using the unique pair_id, get the two experiment IDs from the pair_df
  # table
  exp_ids = dplyr::filter(pair_df, .data$pair_id == !!pair_id)
  exp_ids = dplyr::select(exp_ids, .data$id0, .data$id1)
  exp_ids = unname(c(exp_ids, recursive=T))

  # Using the exp_ids, get the unique experiment settings from the exp_df table
  uniq_settings = dplyr::filter(exp_df, .data$id %in% exp_ids)

  # Finally using the uniq_settings, we need to infer which experiment setting
  # was different (and thus constitute what separates these experiments)
  consider_cols = dplyr::setdiff(colnames(uniq_settings), c("id", "k"))
  nunique = purrr::map_int(.x=consider_cols,
                           .f=~nrow(dplyr::distinct_(uniq_settings, .dots=.x)))

  # Identify the location which has two unique values (the differentiation
  # factor) and then get those values from the DataFrame
  idx = which(nunique == 2)
  vals = dplyr::distinct_(uniq_settings, .dots=consider_cols[idx])
  vals = dplyr::pull(vals)

  # Finally using the unique values and the distinguisher, generate a DataFrame
  # which will help us build a LaTeX table
  return(dplyr::tibble(pair_id=pair_id, distinguisher=consider_cols[idx],
                       setting1=vals[1], setting2=vals[2]))
}

#' Helper function to build the experiment difference DataFrame
#'
#' @param pair_df DataFrame defining the unique experiment pairs
#' @param exp_df DataFrame detailing all experiment settings
#'
#' @return Experiment pair DataFrame
#' @keywords internal
build_exp_diff = function(pair_df, exp_df) {
  uniq_ids = get_uniq_ids(pair_df)
  return(purrr::map_df(.x=uniq_ids, .f=~get_exp_diff(.x, pair_df, exp_df)))
}

#' Generate the experiment difference TeX code
#'
#' @param pair_df DataFrame defining the unique experiment pairs
#' @param exp_df DataFrame detailing all experiment settings
#'
#' @return LaTeX code defining the unique experiments
#'
#' @importFrom rlang .data
#' @keywords internal
gen_exp_setting_tex = function(pair_df, exp_df) {
  # First we need to get each unique pair_id and then using this get all
  # of the unique experiment settings
  setting_df = build_exp_diff(pair_df, exp_df)

  # Update the experiment IDs to be numbers versus the full hashed ID
  setting_df$pair_id = 1:nrow(setting_df)

  # Using the setting_df, generate the TeX code to display each unique
  # experiment
  res = knitr::kable(setting_df, format="latex", caption="Experiment Settings",
                     booktabs=T, digits=3,
                     col.names=c("Experiment #", "Discriminator", "Setting 1",
                                 "Setting 2"))
  return(res)
}

#' Generate all of the LaTeX code we will use to report the results from the
#' bootstrap experiments
#'
#' @param wd Directory containing the files needed to generate the TeX code
#'
#' @return Nothing; saves the .txt files to disk
#' @export
gen_tex_code = function(wd) {
  # Read in the DataFrames from disk
  boot_df = readr::read_csv(file.path(wd, "boot_res.csv"))
  pair_df = readr::read_csv(file.path(wd, "exp_pairs.csv"))
  exp_df = readr::read_csv(file.path(wd, "experiment_settings.csv"))

  # Get the TeX code for the bootstrap results and experiment settings
  boot_tex = gen_boot_res_tex(boot_df)
  exp_tex = gen_exp_setting_tex(pair_df, exp_df)

  # Now we will save the TeX code to disk as .txt files so that they can
  # be copy-pasted later into a report
  dir.create(file.path(wd, "tex_code"), showWarnings = F)
  write(boot_tex, file.path(wd, "tex_code", "boot_res.txt"))
  write(exp_tex, file.path(wd, "tex_code", "exp_diff.txt"))
}
