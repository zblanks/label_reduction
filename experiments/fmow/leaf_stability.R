library(tidyverse)
setwd("c:/devspace/mycode/label_reduction_data/fmow")

# Get the experiment settings and leaf stability DataFrames
settings_df = read_csv("experiment_settings.csv")
stability_df = read_csv("leaf_stability_res.csv")

# Join them so we can have the experiment settings for the appropriate ID
# values
stability_df = stability_df %>% 
  inner_join(settings_df, by="id")

# Show the relationship of entropy values for the flat and hierarchical 
# classifiers
entropy_distn = stability_df %>% 
  filter((metric == "med_entropy") & (label == "all")) %>% 
  ggplot(aes(x=value)) + 
  geom_histogram(aes(fill=method), color="black", bins=30, alpha=0.7,
                 position="identity") + 
  labs(x="Median Entropy Value", y="Count",
       title="Classifier Median Entropy Distribution", fill="Method") + 
  scale_fill_manual(values=c("#d95f02", "#7570b3"), labels=c("Flat", "HC")) +
  theme_gray(base_size=14) +
  theme(legend.position = c(1, 1), legend.justification = c(1, 1),
        legend.background = element_rect(fill=alpha("white", 0.7)),
        legend.key = element_blank())

ggsave("figures/entropy_distn.pdf", entropy_distn, width=8, height=5)

# Show the relationship of log-loss with respect to entropy
entropy_v_log = stability_df %>% 
  spread(key=metric, value=value) %>% 
  filter(label == "all") %>% 
  ggplot(aes(x=med_entropy, y=log_loss)) + 
  geom_point(aes(color=method), size=2) + 
  scale_color_manual(values=c("#d95f02", "#7570b3"), labels=c("Flat", "HC")) + 
  labs(x="Median Entropy", y="Log Loss", color="Method",
       title="Median Entropy vs. Log Loss") + 
  theme_gray(base_size=14) +
  theme(legend.position = c(1, 1), legend.justification = c(1, 1),
        legend.background = element_rect(fill=alpha("white", 0.7)),
        legend.key = element_blank())

ggsave("figures/entropy_log.pdf", entropy_v_log, height=5, width=8)

# Look at the type of arg-max predictions each of the classifiers are making
stability_df %>% 
  filter((label == "all") & (metric == "prob_conf")) %>% 
  group_by(method) %>% 
  summarize(mean(value, na.rm=T))

prob_conf_log = stability_df %>% 
  spread(key=metric, value=value) %>% 
  filter(label == "all") %>% 
  ggplot(aes(x=prob_conf, y=log_loss)) + 
  geom_point(aes(fill=method, size=med_entropy), alpha=0.5, pch=21, 
             color="black") + 
  labs(x="Median Argmax Posterior Probability", y="Log Loss",
       fill="Method", size="Median\nEntropy", 
       title="Classifier Leaf Stabilility Comparison") + 
  scale_fill_manual(values=c("#d95f02", "#7570b3"), labels=c("Flat", "HC")) +
  guides(fill=guide_legend(override.aes = list(size=4))) +
  theme_gray(base_size=14)

ggsave("figures/prob_conf_log.pdf", height=5, width=8)
