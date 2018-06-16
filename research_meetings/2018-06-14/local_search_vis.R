library(tidyverse)
library(stringr)

# Define the working directory for the script
wd = 'c:/devspace/mycode/label_reduction/research_meetings/2018-06-14/'
save_loc = file.path('c:/users/zqb0731/documents/research_writeup/',
                     'research_meetings/2018-06-14/figures/')

# Get the data
df = read_csv(file.path(wd, 'local_search_res.csv'))

# First compare the time to convergence for both neighborhood definitions
df %>% 
  ggplot(aes(x = n_label, y = time)) + 
  geom_line(aes(color = factor(neighborhood_type)), size = 1) + 
  labs(x = 'Number of Labels', y = 'Time (sec)', color = 'Neighborhood Type',
       title = 'Number of Labels vs. Time to Reach Local Maximum')

# Save the first plot to disk
ggsave(file.path(save_loc, 'n_label_vs_time.png'), width = 8, height = 5)

# We want to see how compare the solution quality and also include the time
# to reach the solution
df %>% 
  ggplot(aes(x = n_label, y = obj_val, size = time, 
             fill = neighborhood_type)) +
  geom_point(shape = 21) + 
  labs(x = 'Number of Labels', y = 'Objective Value', size = 'Time (sec)',
       fill = 'Neighborhood Type', 
       title = 'Number of Labels vs. Objective Value')

ggsave(file.path(save_loc, 'n_label_vs_obj_val.png'), width = 8, height = 5)

# Compute the average across all label mappings percent difference in time and
# objective value with complete and one_step neighborhoods
diff_df = df %>% 
  group_by(neighborhood_type) %>% 
  summarize(avg_obj_val = mean(obj_val), avg_time = mean(time))

((diff_df$avg_obj_val[1] - diff_df$avg_obj_val[2])/diff_df$avg_obj_val[2])*100
((diff_df$avg_time[1] - diff_df$avg_time[2])/diff_df$avg_time[2])*100

# Restrict to the interesting number of labels -- L = [15, 35]
diff_df = df %>% 
  filter(n_label %in% 15:35) %>% 
  group_by(neighborhood_type) %>% 
  summarize(avg_obj_val = mean(obj_val), avg_time = mean(time))

(diff_df$avg_obj_val[1] - diff_df$avg_obj_val[2])/diff_df$avg_obj_val[2]
(diff_df$avg_time[1] - diff_df$avg_time[2])/diff_df$avg_time[2]


compute_unbalanced = function(z){
  # Compute how unbalanced the given solution is
  n_label = ncol(z)
  combos = t(combn(1:n_label, 2))
  diff_vals = vector(mode = 'integer', length = nrow(combos))
  for (i in 1:nrow(combos)){
    lab_1 = combos[i, ][1]
    lab_2 = combos[i, ][2]
    diff_vals[i] = abs(sum(z[, lab_1]) - sum(z[, lab_2]))
  }
  
  # Compute how unbalanced the worst case solution is
  m = nrow(z) - n_label + 1
  worst_sum = (m-1)*(n_label-1)
  return(sum(diff_vals)/worst_sum)
}

get_unbalanced_data = function(file){
  # Read in the matrix
  z = as.matrix(read.csv(file, header = F))
  
  # Compute the unbalanced ratio
  unbalance_ratio = compute_unbalanced(z)
  
  # Now we want to extract information from the file so we can create 
  # a DataFrame to use for data visualization
  file = basename(file)
  if (grepl("one_step", file) == TRUE){
    neighborhood_type = "one_step"
  } else{
    neighborhood_type = "complete"
  }
  n_label = ncol(z)
  df = tibble(ratio = unbalance_ratio, neighborhood_type = neighborhood_type,
              n_label = n_label)
  return(df)
}

# Get a vector containing all of the label map files we need to compute
# the difference
map_direc = 'c:/devspace/mycode/label_reduction_data/fruit_maps/'
map_files = list.files(map_direc, full.names = T)

# Get the unbalanced data
unbalance_data = map(.x = map_files, .f = get_unbalanced_data)
unbalance_df = do.call("rbind", unbalance_data)

# Compare the type of solutions we're getting from both one-step and 
# complete neighborhoods in terms of their ratio of the solution found
# relative to the worst case
unbalance_df %>% 
  ggplot(aes(x = n_label, y = ratio, color = neighborhood_type)) + 
  geom_line(size = 1) + 
  labs(x = 'Number of Labels', y = 'Unbalance Ratio',
       color = 'Neighborhood Type',
       title = 'Number of Labels vs. Unbalance Ratio by Neighborhood Type')

ggsave(file.path(save_loc, 'n_label_unbalance.png'), width = 8, height = 5)
