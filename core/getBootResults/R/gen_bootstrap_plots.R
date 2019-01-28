#' Converts the method names into a more read-able format for the plot title
#'
#' @param setting The particular setting for the given experiment
#'
#' @return More read-able string for the given method
#' @keywords internal
convert_method_names = function(setting) {
  # Go through expected cases and adjust the name accordingly
  if (setting == "f") {
    return("Flat Classifier")
  } else if (setting == "hci") {
    return("Hierarchical Classifier")
  }
}

#' Converts the grouping algorithm names
#'
#' @param setting The value for the experiment
#'
#' @return More readable string for the grouping algorithm
#' @keywords internal
convert_group_algo_name = function(setting) {
  if (setting == "kmm") {
    return("K-means Mean")
  } else if (setting == "comm") {
    return("Community Detection")
  }
}

#' Helper function to count the number of unique entries in the specified
#' columns of a DataFrame
#'
#' @param df DataFrame to count the unique entries
#' @param cols Columns to use for counting
#' @param idx Indices to use for counting
#'
#' @return Integer vector containing the number of unique entries in each column
#' @keywords internal
count_uniq_vals = function(df, cols, idx) {
  uniq_vals = purrr::map_int(.x=cols,
                             .f=~dplyr::n_distinct(df[[.x]][idx]))
  return(uniq_vals)
}

#' Helper function to determine if a combination of indices belong together
#'
#' @param exp_diff_df Experiment difference DataFrame
#' @param idx Indices that are being checked for the combination
#'
#' @return Boolean of whether these indices belong together
#' @keywords internal
check_valid_combo = function(exp_diff_df, idx) {
  # A valid combination will have a unique ID and everything else will be
  # the same
  nsamples = length(idx)
  df_cols = colnames(exp_diff_df)
  uniq_vals = count_uniq_vals(exp_diff_df, df_cols, idx)

  # Check if the first index has nsamples unique values and everything else
  # is one
  id_check = uniq_vals[which(df_cols == "pair_id")] == nsamples
  other_check = uniq_vals[which(df_cols != "pair_id")] == 1
  return(id_check && other_check)
}

#' Helper function to determine the other distinguishing factor for a valid
#' combination
#'
#' @param pair_df DataFrame defining the experiment pairs
#' @param exp_df DataFrame detailing all experiment settings
#' @param pair_combo Combination of pair_ids that are valid
#'
#' @return DataFrame containing the pair_combo and the other distinguishing
#' column
#'
#' @importFrom rlang .data
#' @keywords internal
infer_other_col = function(pair_df, exp_df, pair_combo) {
  # Subset the pair_df to just the pair_combo that we need and just
  # grab id0 and pair_id since we don't need id1
  df = dplyr::filter(pair_df, .data$pair_id %in% !!pair_combo)
  df = dplyr::select(df, .data$pair_id, .data$id0)

  # Join the new DataFrame with the experiment settings to get the information
  # to infer the other distinguishing column
  df = dplyr::inner_join(df, exp_df, by=c("id0" = "id"))

  # Excluding the IDs and k, determine the distinguishing column
  ncombos = length(pair_combo)
  df_cols = dplyr::setdiff(colnames(df), c("pair_id", "id0", "k", "run_num"))
  uniq_vals = count_uniq_vals(df, df_cols, 1:nrow(df))

  # The element where all of the entries are different is the other
  # distinguishing column
  other_col = which(uniq_vals == ncombos)
  other_col_name = df_cols[other_col]

  # Using the other_col we will now generate a DataFrame which contains the
  # data needed to extend the plots
  new_df = dplyr::tibble(pair_id=pair_combo, other_col=df[[other_col_name]],
                         other_col_name=other_col_name)
  return(new_df)
}

#' Infers if any of the experiments belong as a group for plotting purposes
#'
#' @param exp_diff_df Experiment difference DataFrame
#' @param exp_df DataFrame detailing all experiment settings
#' @param pair_df DataFrame defining the experiment pairs
#'
#' @return List containing the pair_ids that belong togeter and a DataFrame
#' used for plotting
#' @keywords internal
infer_exp_combos = function(exp_diff_df, exp_df, pair_df) {
  # First we need to know the number of unique experiment pairs because this
  # will define the maximum number of groups
  nsamples = nrow(exp_diff_df)

  # Get the vector of all pair_ids to add to the pair_id combos
  uniq_ids = get_uniq_ids(exp_diff_df)

  # Define empty lists to hold the resulting DataFrames ID combos
  id_combos = list()
  combo_dfs = list()
  count = 0

  # Go through the power set of combinations and infer if any of them
  # belong with one another
  for (i in nsamples:1) {
    # Generate the choose(n, k) combinations for this value of i
    combo_idx = utils::combn(nsamples, i)

    # Go through each idx combination and check if that combo belongs
    # together
    for (j in 1:ncol(combo_idx)) {
      tmp_idx = combo_idx[, j]
      is_valid_combo = check_valid_combo(exp_diff_df, tmp_idx)

      # Check if all of the pair IDs are still available
      pair_combo = exp_diff_df$pair_id[tmp_idx]
      all_available = all(pair_combo %in% uniq_ids)

      # If the combination is valid and the IDs are still present then add them
      # to the id_combos list and infer the other distinguishing factor
      # for the DataFrame
      if ((is_valid_combo == TRUE) && (all_available == TRUE)) {
        # Add the pair IDs to the list
        count = count + 1
        id_combos[[count]] = pair_combo

        # Remove the pair_combo as a feasible choice from all the vectors
        uniq_ids = uniq_ids[-which(uniq_ids %in% pair_combo)]

        # Generate a DataFrame that infers what also distinguished this group
        if (length(tmp_idx) == 1) {
          combo_dfs[[count]] = dplyr::tibble(pair_id=exp_diff_df$pair_id[tmp_idx])
        } else {
          combo_dfs[[count]] = infer_other_col(pair_df, exp_df, pair_combo)
        }
      }
    }
  }

  # Return a list containing the id_combos and combo_dfs
  return(list("ids" = id_combos, "dfs" = combo_dfs))
}

#' Helper function to get the setting values for a given experiment
#'
#' @param exp_diff_df Experiment difference DataFrame
#'
#' @return Character vector of experiment settings
#' @keywords internal
get_experiment_settings = function(exp_diff_df) {
  settings = dplyr::select(exp_diff_df[1, ], dplyr::starts_with("setting"))
  settings = unname(c(settings, recursive=T))
  return(settings)
}

#' Helper function to generate the plot and caption title based off the
#' experiment distinguishing factor
#'
#' @param exp_diff_df Experiment difference DataFrame
#'
#' @return List containing the plot and caption title
#' @keywords internal
gen_plot_titles = function(exp_diff_df) {
  # First we have to get the unique settings for the experiment
  settings = get_experiment_settings(exp_diff_df)

  # Depending on what the distinguisher is, we have to adjust the settings
  # accordingly
  if (exp_diff_df$distinguisher[1] == "method") {
    settings = purrr::map_chr(.x=settings, .f=convert_method_names)
  } else if (exp_diff_df$distinguisher[1] == "group_algo") {
    settings = purrr::map_chr(.x=settings, .f=convert_group_algo_name)
  }

  # Using the updated names, generate the plot and caption titles
  plot_title = paste(settings[2], "vs.", settings[1], "Bootstrap Distributions")
  caption_title = paste("Percent difference calculated with respect to",
                        settings[2])

  return(list("plot_title" = plot_title, "caption_title" = caption_title))
}

#' Helper function generate the facet names
#' @return Named vector of metrics
#' @keywords internal
gen_facet_names = function() {
  return(c("leaf_top1" = "Leaf Top 1", "leaf_top3" = "Leaf Top 3",
           "node_top1" = "Node Top 1"))
}

#' Function to generate a plot under the condition that there is only one
#' pair_combo that is valid
#'
#' @param pair_combo Lone pair_id
#' @param boot_df Bootstrap results DataFrame
#' @param figure_titles List containing the plot and caption title
#'
#' @return ggplot for bootstrap histograms
#'
#' @importFrom rlang .data
#' @keywords internal
gen_lone_hist = function(pair_combo, boot_df, figure_titles) {
  # Since we only have one pair_combo we only need to re-name the facets for
  # for the evaluation metrics
  facet_names = gen_facet_names()

  # Subset the DataFrame and generate the plot
  new_df = dplyr::filter(boot_df, .data$pair_id == !!pair_combo)
  p = ggplot2::ggplot(new_df, ggplot2::aes(x=.data$value, group=.data$metric)) +
    ggplot2::geom_histogram(color="white") +
    ggplot2::facet_wrap(~metric, scales="free",
                        labeller=ggplot2::labeller(metric=facet_names)) +
    ggplot2::labs(x="Percent Difference", y="Count",
                  title=figure_titles$plot_title,
                  caption=figure_titles$caption_title) +
    ggplot2::theme_bw()

  return(p)
}

#' Function to generate a plot under the condition that there are two
#' distinguishing factors for an experiment (ex: method and estimator)
#'
#' @param pair_combo Combination of valid pair_ids
#' @param boot_df Bootstrap results DataFrame
#' @param update_df DataFrame which adds the information we need to extend
#' the plot
#' @param figure_titles List containing the plot and caption title
#'
#' @return ggplot for bootstrap histograms
#'
#' @importFrom rlang .data
#' @keywords internal
gen_combo_hist = function(pair_combo, boot_df, update_df, figure_titles) {
  # First we need to update the bootstrap DataFrame to include the other
  # distinguisher as well as the specific pair_id combo
  df = dplyr::filter(boot_df, .data$pair_id %in% pair_combo)
  df = dplyr::inner_join(df, update_df, by="pair_id")

  # Next we need to generate the facet grid update based on what
  metric_names = gen_facet_names()
  if (df$other_col_name[1] == "estimator") {
    other_names = c("log" = "LR", "rf" = "RF")
  }

  # Finally generate the plot
  p = ggplot2::ggplot(df, ggplot2::aes(x=.data$value, group=.data$metric)) +
    ggplot2::geom_histogram(color="white") +
    ggplot2::facet_grid(other_col~metric, scales="free_x",
                        labeller=ggplot2::labeller(metric=metric_names,
                                                   other_col=other_names)) +
    ggplot2::labs(x="Percent Difference", y="Count",
                  title=figure_titles$plot_title,
                  caption=figure_titles$caption_title) +
    ggplot2::theme_bw()

  return(p)
}

#' Creates and saves the bootstrap histograms to disk
#'
#' @param pair_combo Unique experiment pair ID
#' @param update_df DataFrame which could add new data to help extend the plot
#' @param boot_df Bootstrap results DataFrame
#' @param exp_diff_df Experiment difference DataFrame
#' @param savepath Directory in which to save the plot
#'
#' @return Nothing; saves the plot to disk
#'
#' @importFrom rlang .data
#' @keywords internal
gen_boot_hists = function(pair_combo, update_df, boot_df, exp_diff_df,
                          savepath) {
  # Filter the experiment DataFrame to contain the correct pair_ids
  new_exp_diff_df = dplyr::filter(exp_diff_df, .data$pair_id %in% !!pair_combo)

  # Using the experiment differences, we need to generate the correct plot
  # and caption title
  figure_titles = gen_plot_titles(new_exp_diff_df)

  # We need to generate the plot differently depending on how many elements
  # are in the pair_combo
  ncombos = length(pair_combo)

  # Get the base names for the filepath
  plot_base = new_exp_diff_df$distinguisher[1]
  settings = get_experiment_settings(new_exp_diff_df)

  if (ncombos == 1) {
    p = gen_lone_hist(pair_combo, boot_df, figure_titles)

    # Generate the meaningful save name
    figure_name = paste(plot_base, '_', settings[1], '_vs_', settings[2],
                        '.pdf', sep='')
  } else {
    p = gen_combo_hist(pair_combo, boot_df, update_df, figure_titles)

    # Generate a save name with the other factor
    figure_name = paste(plot_base, '_', settings[1], '_vs_', settings[2],
                        '_plus_', update_df$other_col_name[1], '.pdf', sep='')
  }

  # Save the plot to disk
  pair_id = paste(pair_combo, collapse="")
  filepath = file.path(savepath, "figures", figure_name)
  ggplot2::ggsave(filepath, plot=p, scale=0.6, height=(4 * ncombos), width=12)
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
  exp_diff_df = build_exp_diff(pair_df, exp_df)

  # Get the pair combos and their corresponding DataFrames
  res = infer_exp_combos(exp_diff_df, exp_df, pair_df)

  # Go through each unique pair_id in the exp_diff and generate and save
  # the plot to disk
  dir.create(file.path(wd, "figures"), showWarnings=F)
  purrr::map2(.x=res$ids, .y=res$dfs, .f=~gen_boot_hists(.x, .y, boot_df,
                                                         exp_diff_df, wd))
}
