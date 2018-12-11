library(scales)
source("c:/devspace/mycode/label_reduction/bootstrap_results.R")

# Get the appropriate data
wd = "c:/devspace/mycode/label_reduction_data/fmow"
df = read_csv(file.path(wd, "leaf_root_res.csv"))
settings = read_csv(file.path(wd, "experiment_settings.csv"))
df = df %>% inner_join(settings, by="id")

# Define the metrics we are interested in computing
metrics = unique(df$metric)
use_meta_vals = unique(df$use_meta)
val_grid = expand.grid(metric=metrics, use_meta=use_meta_vals)
val_grid$metric = as.character(val_grid$metric)

res = map2(.x=val_grid$metric, .y=val_grid$use_meta,
           .f = ~ get_boot_distn(df, metric=.x, nsamples=10000, use_meta=.y))


# Go through each of the data sets and add the releveant experiment
# settings to the data so that we can make an appropriate plot title
get_experiment_settings = function(boot_dict, settings, boot_res) {
  # Update the boot_res DataFrame by adding the experiment settings
  new_boot_res = boot_dict %>% 
    inner_join(select(settings, id, use_meta), by="id") %>% 
    distinct(boot_id, use_meta) %>% 
    inner_join(boot_res, by="boot_id") %>% 
    mutate(value = -value)
  
  return(new_boot_res)
}

# Update the boot_res DataFrames
boot_dicts = res %>% map(~ .x$boot_id_dict)
boot_results = res %>% map(~ .x$boot_res)
boot_res_df = map2_df(.x=boot_dicts, .y=boot_results,
                      .f=~ get_experiment_settings(.x, settings, .y))

# Plot the results for all of the experiment settings and metric values
plot_res = function(df, metric, wd) {
  # Subset the data on the appropriate metric
  df = df %>% filter(metric == UQ(metric))
  
  # Clean up the metric string so it can be used in the title of the plot
  metric = str_replace(metric, "_", " ")
  metric = str_to_title(metric)
  
  if (str_detect(metric, "Auc")) {
    metric = str_replace(metric, "Auc", "AUC")
  } else {
    number = str_extract(metric, "[0-9]")
    metric = str_replace(metric, "[0-9]", "")
    metric = paste(metric, number, "Accuracy")
  }
  
  # Make the title string
  title_str = paste(metric, "Distribution")
  
  # Define the legend guide
  guide = guide_legend(title="Use\nMeta-Data", title.position="top",
                       keyheight = unit(1.25, "cm"), 
                       keywidth = unit(1.25, "cm"))
  
  # Make the plot
  df %>% 
    ggplot(aes(x=value)) +
    geom_histogram(aes(fill=factor(use_meta)), alpha=0.5, color="black") + 
    labs(x="Percent Increase", y="Count", title=title_str) + 
    scale_fill_manual(values=c("#d95f02", "#7570b3"), guide=guide,
                      labels=c("No", "Yes")) + 
    theme_grey(base_size = 50) + 
    theme(legend.position = c(1, 1), legend.justification = c(1, 1),
          legend.background = element_rect(fill=alpha("white", 0.7)),
          legend.key = element_blank())
  
  # Save the plot to disk
  metric_str = df %>% distinct(metric) %>% pull
  path = file.path(wd, paste(metric_str, ".pdf", sep=""))
  ggsave(path, height=20, width=24.27)
}

savepath = "c:/users/zqb0731/documents/research_writeup/meetings/2018-11-02"
metrics %>% map(~ plot_res(boot_res_df, .x, savepath))
