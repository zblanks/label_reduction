library(tidyverse)
setwd("c:/devspace/mycode/label_reduction_data/fmow")
settings_df = read_csv("experiment_settings.csv")
search_df = read_csv("search_res.csv")

# Get the experiment settings
search_df = search_df %>% 
  inner_join(settings_df, by="id")

# Basic summary statistics to comapre the two search methods
search_df %>% 
  group_by(metric, group_algo) %>% 
  summarize(mean(value))

# Visualize the training time for each algorithm as well as the performance
# distribution
group_algo_time_compare = search_df %>% 
  filter(metric == "train_time") %>% 
  group_by(k, group_algo) %>% 
  summarize(med_train_time = median(value)) %>% 
  ggplot(aes(x=k, y=med_train_time)) + 
  geom_line(aes(color=group_algo), size=2) +
  scale_color_manual(values=c("#d95f02", "#7570b3"), labels=c("CD", "KMM")) + 
  labs(x="Number of Meta-Classes", y="Median Validation Training Time",
       color="Grouping Heuristic", 
       title="Label Grouping Training Time Comparison") + 
  theme_gray(base_size=14) +
  theme(legend.position = c(1, 1), legend.justification = c(1, 1),
        legend.background = element_rect(fill=alpha("white", 0.7)),
        legend.key = element_blank())

ggsave("figures/group_algo_time_compare.pdf", height=5, width=8)  

# Compare the validation metric performance between the two grouping algorithms
group_algo_perf_compare = search_df %>% 
  filter((metric == "top1") & (use_meta == 1)) %>% 
  group_by(k, group_algo) %>% 
  summarize(med_top1 = median(value)) %>% 
  ggplot(aes(x=k, y=med_top1)) + 
  geom_point(aes(color=group_algo), size=2) + 
  scale_color_manual(values=c("#d95f02", "#7570b3"), labels=c("CD", "KMM")) + 
  labs(x="Number of Meta-Classes", y="Median Validation Leaf Top 1",
       color="Grouping Heuristic", 
       title="Label Grouping Performance Comparison") + 
  theme_gray(base_size=14) +
  theme(legend.position = c(1, 1), legend.justification = c(1, 1),
        legend.background = element_rect(fill=alpha("white", 0.7)),
        legend.key = element_blank())

ggsave("figures/group_algo_perf_compare.pdf", group_algo_perf_compare,
       height=5, width=8)
