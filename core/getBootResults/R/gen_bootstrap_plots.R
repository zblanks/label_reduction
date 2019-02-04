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
  } else if (setting == 'hci_kmm') {
    return('Hierarchical Classifier with K-Means Mean')
  } else if (setting == 'hci_comm') {
    return('Hierarchical Classifier with Community Detection')
  } else if (setting == 'hci_lp') {
    return('Hierarchical Classifier with LP Heuristic')
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
  } else if (setting == 'lp') {
    return("LP Heuristic")
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
  return(id_check & all(other_check))
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
#' @param max_combo_size Max combination size for experiment combos
#'
#' @return List containing the pair_ids that belong togeter and a DataFrame
#' used for plotting
#' @keywords internal
infer_exp_combos = function(exp_diff_df, exp_df, pair_df, max_combo_size) {
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
  for (i in max_combo_size:1) {
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
#' @return Plot title
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
  return(plot_title)
}

#' Helper function generate the facet names
#' @return Named vector of metrics
#' @keywords internal
gen_facet_names = function() {
  return(c("leaf_top1" = "Leaf Top 1", "leaf_top3" = "Leaf Top 3",
           "node_top1" = "Node Top 1"))
}

#' Helper function to prepare the bootstrap DataFrame for plotting
#'
#' @param boot_df Bootstrap results DataFrame
#' @param info_df DataFrame containing the plotting info for the \code{boot_df}
#'
#' @return Updated DataFrame
#'
#' @importFrom rlang .data
#' @keywords internal
prep_boot_df = function(boot_df, info_df) {
  # Grab the distinct IDs
  exp_ids = dplyr::pull(dplyr::distinct(info_df, .data$exp_id))

  # First we need to update the bootstrap DataFrame to include the other
  # distinguisher as well as the specific pair_id combo
  df = dplyr::filter(boot_df, .data$exp_id %in% exp_ids)

  # Add the plotting information
  df = dplyr::inner_join(df, info_df, by='exp_id')
  return(df)
}

#' Helper function to create a legend labeller
#'
#' @param distinguisher Distinguishing factor for the experiment
#'
#' @return Named vector to map the legend labels
#' @keywords internal
build_legend_labeller = function(distinguisher) {
  # Update the values for the legend based on the distinguisher
  if (distinguisher == 'method') {
    labels = c('f' = 'FC', 'hci_kmm' = 'HC w/ KMM', 'hci_lp' = 'HC w/ LP',
               'hci_comm' = 'HC w/ CD')
  } else if (distinguisher == 'group_algo') {
    labels = c('kmm' = 'KMM', 'comm' = 'CD', 'lp' = 'LP')
  }

  return(labels)
}

#' Function to generate a plot under the condition that there is only one
#' pair_combo that is valid
#'
#' @param boot_df Bootstrap results DataFrame
#' @param info_df DataFrame containing the plotting info for the \code{boot_df}
#' @param plot_title The plot title
#'
#' @return ggplot for bootstrap distributions
#'
#' @importFrom rlang .data
#' @keywords internal
gen_lone_distn = function(boot_df, info_df, plot_title) {
  # Add the additional information for plotting
  df = prep_boot_df(boot_df, info_df)

  # Since we only have one pair_combo we only need to re-name the facets for
  # for the evaluation metrics
  facet_names = gen_facet_names()

  # We don't need a legend title since the info is already provided in the
  # plot title
  guide = ggplot2::guide_legend(title='')
  labels = build_legend_labeller(df$distinguisher[1])

  # Subset the DataFrame and generate the plot
  p = ggplot2::ggplot(df, ggplot2::aes(x=.data$value, color=.data$setting)) +
    ggplot2::geom_density() +
    ggplot2::facet_wrap(~metric, scales="free",
                        labeller=ggplot2::labeller(metric=facet_names)) +
    ggplot2::labs(x="Value", y="Density", title=plot_title) +
    ggplot2::scale_color_manual(values=c("#d95f02", "#7570b3"), guide=guide,
                                labels=labels) +
    ggplot2::theme_bw()

  return(p)
}

#' Function to generate a plot under the condition that there are two
#' distinguishing factors for an experiment (ex: method and estimator)
#'
#' @param boot_df Bootstrap results DataFrame
#' @param info_df DataFrame containing the plotting info for the \code{boot_df}
#' @param plot_title The plot title
#'
#' @return ggplot for bootstrap distributions
#'
#' @importFrom rlang .data
#' @keywords internal
gen_combo_distn = function(boot_df, info_df, plot_title) {
  # Prep the bootstrap DataFrame with the other info
  df = prep_boot_df(boot_df, info_df)

  # Next we need to generate the facet grid update based on what
  metric_names = gen_facet_names()
  if (df$other_col_name[1] == "estimator") {
    other_names = c("log" = "LR", "rf" = "RF")
  }

  # Update the values for the legend based on the distinguisher
  labels = build_legend_labeller(df$distinguisher[1])
  labels = build_legend_labeller(df$distinguisher[1])

  # We don't need a legend title since the info is already provided in the
  # plot title
  guide = ggplot2::guide_legend(title='')

  # Finally generate the plot
  p = ggplot2::ggplot(df, ggplot2::aes(x=.data$value, color=.data$setting)) +
    ggplot2::geom_density() +
    ggplot2::facet_grid(other_col~metric, scales="free",
                        labeller=ggplot2::labeller(metric=metric_names,
                                                   other_col=other_names)) +
    ggplot2::labs(x="Value", y="Density", title=plot_title) +
    ggplot2::scale_color_manual(values=c("#d95f02", "#7570b3"), guide=guide,
                                labels=labels) +
    ggplot2::theme_bw()

  return(p)
}

#' Helper function to deal with columns like setting1, setting2 and
#' convert them into a single vector
#'
#' @param df DataFrame containing the columns
#' @param identifier Identifier that distinguishes the column
#'
#' @return Vector which flattens all the entries
#'
#' @keywords internal
combine_like_columns = function(df, identifier) {
  return(unname(unlist(dplyr::select(df, dplyr::starts_with(identifier)))))
}

#' Creates and saves the bootstrap distributions to disk
#'
#' @param pair_combo Unique experiment pair ID
#' @param update_df DataFrame which could add new data to help extend the plot
#' @param boot_df Bootstrap results DataFrame
#' @param exp_diff_df Experiment difference DataFrame
#' @param pair_df DataFrame defining the experiment pairs
#' @param exp_df DataFrame detailing all experiment settings
#' @param savepath Directory in which to save the plot
#'
#' @return Nothing; saves the plot to disk
#'
#' @importFrom rlang .data
#' @keywords internal
gen_boot_distns = function(pair_combo, update_df, boot_df, exp_diff_df, pair_df,
                          exp_df, savepath) {
  # Filter the experiment DataFrame to contain the correct pair_ids
  exp_diff_df = dplyr::filter(exp_diff_df, .data$pair_id %in% !!pair_combo)

  # Using the experiment differences, we need to generate the correct plot
  # and caption title
  plot_title = gen_plot_titles(exp_diff_df)

  # We need to generate the plot differently depending on how many elements
  # are in the pair_combo
  ncombos = length(pair_combo)

  # Get the base names for the filepath
  plot_base = exp_diff_df$distinguisher[1]
  settings = get_experiment_settings(exp_diff_df)

  # Grab all of the experiment IDs that belong with the pair_combo
  pair_df = dplyr::filter(pair_df, .data$pair_id %in% !!pair_combo)
  exp_ids = combine_like_columns(pair_df, 'id')

  # We need a DataFrame that will allow us to add the necessary data to
  # the bootstrap DataFrame and we need to do this by combining the
  # experiment IDs
  info_df = dplyr::tibble(exp_id=exp_ids, pair_id=rep(pair_df$pair_id, 2))

  # Using the updated pair_df we now need to add the distinguishing and
  # updating information so we can combine this with the boot DataFrame
  setting_vals = combine_like_columns(exp_diff_df, 'setting')
  info_df = dplyr::mutate(info_df, setting=setting_vals,
                          distinguisher=exp_diff_df$distinguisher[1])
  info_df = dplyr::inner_join(info_df, update_df, by='pair_id')
  info_df = dplyr::select(info_df, -pair_id)

  if (ncombos == 1) {
    p = gen_lone_distn(exp_ids, boot_df, plot_title)

    # Generate the meaningful save name
    figure_name = paste(plot_base, '_', settings[1], '_vs_', settings[2],
                        '.pdf', sep='')
  } else {
    p = gen_combo_distn(boot_df, info_df, plot_title)

    # Generate a save name with the other factor
    figure_name = paste(plot_base, '_', settings[1], '_vs_', settings[2],
                        '_plus_', update_df$other_col_name[1], '.pdf', sep='')
  }

  # Save the plot to disk
  pair_id = paste(pair_combo, collapse="")
  filepath = file.path(savepath, "figures", figure_name)
  R.devices::suppressGraphics(ggplot2::ggsave(filepath, plot=p, scale=0.7,
                                              height=(4 * ncombos), width=12))
}

#' Generates all of the bootstrap distributions
#'
#' @param wd Directory containing the files needed to generate the plots
#' @param max_combo_size Max combination size for experiment combos
#'
#' @return Nothing; saves the plots to disk
#' @export
gen_boot_plots = function(wd, max_combo_size=2) {
  # Get the bootstrap results as well as the files needed to get the
  # unique experiments
  boot_df = readr::read_csv(file.path(wd, "boot_res.csv"))
  pair_df = readr::read_csv(file.path(wd, "exp_pairs.csv"))
  exp_df = readr::read_csv(file.path(wd, "experiment_settings.csv"))

  # Get the experiment differences
  exp_diff_df = build_exp_diff(pair_df, exp_df)

  # Get the pair combos and their corresponding DataFrames
  res = infer_exp_combos(exp_diff_df, exp_df, pair_df, max_combo_size)

  # Go through each unique pair_id in the exp_diff and generate and save
  # the plot to disk
  dir.create(file.path(wd, "figures"), showWarnings=F)
  purrr::map2(.x=res$ids, .y=res$dfs, .f=~gen_boot_distns(.x, .y, boot_df,
                                                          exp_diff_df, pair_df,
                                                          exp_df, wd))
}
