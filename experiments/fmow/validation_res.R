library(tidyverse)
library(wordcloud)
library(scales)

# Get the experiment settings and search results
setwd("c:/devspace/mycode/label_reduction_data/fmow")
savewd = "c:/users/zqb0731/documents/research_writeup/meetings/2018-11-09"
settings = read_csv("experiment_settings.csv")
search = read_csv("search_res.csv")

#### HC Algorithmic Performance ####

# Join the tables so we have the experiment settings
search = search %>% inner_join(settings, by = "id")

# First let's take a look at how the top1 accuracy changes as we shift k
# for both cases of using and not using the meta-data
search %>%
  filter(metric == "top1") %>% 
  group_by(k, use_meta) %>% 
  summarize(value = mean(value)) %>% 
  ggplot(aes(x=k, y=value, color=factor(use_meta))) + 
  geom_line(size=2) + 
  scale_color_manual(values=c("#d95f02", "#7570b3")) + 
  labs(x="Number of Meta-Classes", y="Average Leaf Top 1 Accuracy",
       color="Used\nMeta-Data", 
       title="Number of Meta-Classes vs. Leaf Top 1 Accuracy") + 
  theme_grey(base_size=14)

ggsave(file.path(savewd, "k_cv.pdf"), height=5, width=8)

# Let's count how many times a given value of k was considered the "best"
# for a given dataset
get_best_k = function(df, run_num, use_meta, metric) {
  # Get the best value of k
  best_k = df %>% 
    filter((run_num == UQ(run_num)) & (use_meta == UQ(use_meta)) & 
             (metric == UQ(metric))) %>% 
    slice(which.max(value)) %>% 
    pull(k)
  
  # Get the final DataFrame
  new_df = tibble(run_num=run_num, use_meta=use_meta, metric=metric,
                  best_k=best_k)
  return(new_df)
}

# Get the best_k DataFrame
val_grid = expand.grid(run_num=unique(search$run_num), 
                       use_meta=unique(search$use_meta),
                       metric=setdiff(unique(search$metric), c("train_time")))
val_grid = list(run_num=val_grid$run_num, use_meta=val_grid$use_meta,
                metric=val_grid$metric)
val_grid$metric = as.character(val_grid$metric)

best_k_df = val_grid %>% pmap_df(~ get_best_k(search, ..1, ..2, ..3))

# Visualize the distribution of best k values for top1 and AUC
best_k_df %>% 
  ggplot(aes(x=best_k)) + 
  facet_grid(vars(metric), vars(use_meta)) + 
  geom_histogram(bins=59, aes(y=(..count..) / (sum(..count..)))) + 
  labs(x="Best k", y="Fraction of Runs")

# Let's grab the FC training time results so we can compare how expensive
# it is to train each model
fc_res = read_csv("fc_prelim_res.csv")
median_fc_train_time = fc_res %>% 
  filter(metric == "train_time") %>% 
  pull(value) %>%
  median

guide = guide_legend(title="Use\nMeta-Data", title.position="top",
                     keyheight = unit(1.25, "cm"), 
                     keywidth = unit(1.25, "cm"))

# See how the training time is affected by changing k
search %>% 
  filter(metric == "train_time") %>%
  group_by(k, use_meta) %>% 
  summarize(train_time = median(value)) %>% 
  ggplot(aes(x=k, y=train_time, color=factor(use_meta))) + 
  geom_line(size=4) + 
  scale_color_manual(values=c("#d95f02", "#7570b3"), guide=guide,
                     labels=c("No", "Yes")) + 
  geom_hline(yintercept=median_fc_train_time, linetype="dashed", size=2) + 
  annotate("text", label="Median Flat Classifier Training Time", x=35, 
           y=median_fc_train_time + 1, fontface=2, size=14) + 
  labs(x="Number of Meta-Classes", y="Median Training Time (sec)",
       title="Number of Meta-Classes vs. Median Training Time") + 
  theme_grey(base_size=50) + 
  theme(legend.position = c(1, 1), legend.justification = c(1, 1),
        legend.background = element_rect(fill=alpha("white", 0.7)),
        legend.key = element_blank())

ggsave(file.path(savewd, "k_train_time.pdf"), height=15, width=24.27)

#### Inferred Meta-Classes ####

# Get the meta-class data
group = read_csv("group_res.csv")
group = group %>% inner_join(settings, by="id")

# Get the final clustering of the meta-class data
label_clustering = read_csv("label_clustering.csv")

# Build a DataFrame which maps the label to the their name (ex: airport -> 0)
label_names = list.files("d:/fmow_rgb/train")
label_df = tibble(label=0:(length(label_names) - 1), label_names=label_names)

# Add the names to the label_clustering DF
label_clustering = label_clustering %>% inner_join(label_df, by="label")

# Create the directories to hold the results for each k setting
k_vals = label_clustering %>% distinct(k) %>% pull(k)
for (i in 1:length(k_vals)) {
  dir.create(file.path(savewd, "wordclouds", k_vals[i]))
}

# Plots the wordcloud for a given k and group value
plot_wordcloud = function(df, k, group, use_meta, savewd) {
  # Subset the data on the appropriate value of k and group
  df = df %>%
    filter((k == UQ(k)) & (group == UQ(group)) & (use_meta == UQ(use_meta)))
  
  # Convert the data into a word frequency table to be used for the wordcloud
  words = df %>% pull(label_names)
  freqs = rep(1, length(words))
  wordcloud_df = tibble(word=words, freq=freqs)
  
  # Save the wordcloud to disk
  path = file.path(savewd, "wordclouds", k, 
                   paste("wc_g", group, "_k", k, "_um", use_meta, 
                         ".pdf", sep=""))
  set.seed(17)
  pdf(file=path, height=5, width=8)
  
  # Determine the appropriate scale for the words
  if (nrow(wordcloud_df) <= 10) {
    scale = c(4, .5)
  } else if (nrow(wordcloud_df) <= 20) {
    scale = c(2, .25)
  } else {
    scale = c(1, .12)
  }
  
  wordcloud(words=wordcloud_df$word, freq=wordcloud_df$freq,
            random.color=T, colors=c("#a6cee3", "#1f78b4", "#b2df8a",
                                     "#33a02c", "#fb9a99", "#e31a1c"),
            scale=scale)
  dev.off()
}

# Define all of the combinations of k, group, and use_meta to plot the
# respective word clouds
val_grid = label_clustering %>% distinct(k, group, use_meta)
pmap(.l=list(val_grid$k, val_grid$group, val_grid$use_meta),
     .f=~plot_wordcloud(label_clustering, ..1, ..2, ..3, savewd))

# Get the clustering similiarty data
cluster_res = read_csv("cluster_sim_res.csv")

# Get the summary data for each experiment setting, value for k, and metric
cluster_res %>% 
  filter(metric != "jacard") %>% 
  group_by(k, use_meta, metric) %>% 
  summarize(avg_val = mean(value)) %>% 
  ggplot(aes(x=k, y=avg_val, color=metric)) + 
  geom_line(size=2) + 
  facet_grid(vars(use_meta))
