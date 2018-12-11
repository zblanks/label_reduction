library(tidyverse)
library(digest)

boot_mean = function(x, nsamples) {
  #' Computes the mean of a bootstrapped vector
  #' 
  #' @param x (vector; double): Vector of values used for the bootstrap
  #' @param nsamples (int): Number of bootstrap samples
  #' 
  #' @return Mean of bootstrapped vector
  
  # Generate the bootstrap samples
  n = length(x)
  x_list = rep(list(x), nsamples)
  boot_samples = x_list %>% map(~ sample(.x, n, replace = T))
  
  # Compute the mean of each bootstrapped sample
  return(boot_samples %>% map_dbl(mean))
}

mean_diff = function(df, metric, nsamples, ids) {
  #' Computes the mean percent difference between two methods
  #' 
  #' @param df (tibble): Tibble containing the experiment results
  #' @param metric (character): Metric we are interested in comparing
  #' @param nsamples (int): Number of bootstrap samples
  #' @param ids (vector; character): Vector of IDs for a given set of experiment values
  #' 
  #' @return Bootstrap distribution of mean difference
  
  # Grab the vectors that correspond to the metric and ID values
  method_values = ids %>% map(~ df %>% filter(id %in% .x) %>% pull(value))
  
  # Compute the bootstrapped mean for the given methods
  boot_vals = method_values %>% map(~ boot_mean(.x, nsamples))
  mean_perc_diff = ((boot_vals[[2]] - boot_vals[[1]]) / (boot_vals[[1]])) * 100
  return(mean_perc_diff)
}

get_ids = function(df, methods) {
  #' Grabs the IDs that correspond to a given pair of methods
  #' 
  #' @param df (tibble): DataFrame containing experiment results
  #' @param methods (vector; character): Methods we want to subset
  #' 
  #' @return Vector of IDs from the DataFrame

  # Grabs the IDs that correspond to the given pair of methods
  ids = unlist(methods) %>% map(~ df %>% filter(method == UQ(.x)) %>% pull(id))
  return(ids)
}

format_boot_results = function(boot_id, boot_distn, metric) {
  #' Formats the bootstrap results in a "tidy" way
  #' 
  #' @param boot_id (character) Bootstrap iteration ID
  #' @param boot_distn (vector; double): Distribution from bootstrap results
  #' @param metric (character): Metric we computed
  #' 
  #' @return "tidy" bootstrap DataFrame
  
  nsamples = length(boot_distn)
  sample_num = 1:nsamples
  
  # Build the tibble for the bootstrap results
  boot_res = tibble(boot_id=rep(boot_id, nsamples), sample_num=sample_num,
                    metric=rep(metric, nsamples), value=boot_distn)
  
  return(boot_res)
}

build_boot_id_dict = function(boot_id, ids) {
  #' Builds the boot_id "dictionary"
  #' 
  #' @param boot_id (character): Bootstrap iteration ID
  #' @param ids (vector; character): Vector of IDs for a given set of experiment values
  #' 
  #' @return "dictionary" to map boot_ids
  
  # Unlist the ID values because we need them to be a vector
  ids = unlist(ids)
  
  # Format the DataFrame
  n = length(ids)
  method_type = rep(c(1, 2), each=(n / 2))
  return(tibble(boot_id=rep(boot_id, n), id=ids, method_type=method_type))
}

get_boot_distn = function(df, metric, nsamples=10000, ...) {
  #' Computes the bootstrap distribution for a given metric and experiment settings
  #' 
  #' @param df (tibble): Tibble containing the experiment results
  #' @param metric (character): Metric we're interested in computing
  #' @param nsamples (int): Number of bootstrap samples
  #' 
  #' @return "tidy" bootstrap results for the given metric
  
  # Grab the IDs that correspond to the given experiment settings
  settings = list(...)
  var_names = names(settings)
  var_vals = unlist(settings, use.names=F)
  df = df %>% 
    filter_at(vars(var_names), any_vars(. == var_vals)) %>% 
    filter(metric == UQ(metric))
  
  # Given the filtered DataFrame, we will now infer the pairwise ID
  # values for each combination
  unique_methods = unique(df$method)
  method_combos = combn(unique_methods, 2)
  method_combos = apply(method_combos, 2, list) %>% map(unlist)
  combo_ids = method_combos %>% map(~ get_ids(df, .x))
  
  # Compute the bootstrap distribution for each of the method combinations
  boot_distns = combo_ids %>% map(~ mean_diff(df, metric, nsamples, .x))
  
  # Build the IDs for the given bootstrap experiment
  boot_ids = method_combos %>% 
    map(~ paste(.x, var_vals, sep="", collapse="")) %>% 
    map(sha1)
  
  # Build the DataFrames for each of the bootstrap distributions
  boot_res = map2_df(.x=boot_ids, .y=boot_distns, 
                     .f = ~ format_boot_results(.x, .y, metric))
  
  # Build the "dictionary" for the given boot_id to map the experiment
  # IDs
  boot_id_dict = map2_df(.x=boot_ids, .y=combo_ids,
                         .f = ~ build_boot_id_dict(.x, .y))
  
  return(list("boot_res" = boot_res, "boot_id_dict" = boot_id_dict))
}
