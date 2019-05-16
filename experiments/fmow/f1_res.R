library(tidyverse)
setwd('c:/devspace/mycode/label_reduction_data/fmow')
df = read_csv('f1_res.csv')

df = df %>% 
  gather(key=label, value=f1, starts_with("f1"), -method, -estimator) %>% 
  mutate(label = as.numeric(str_replace(label, "f1_", "")))

estimator_names = c("log" = "LR", "rf" = "Random Forest", 'knn' = 'KNN')

p = df %>% 
  filter((method %in% c("FC", "KMC") & (estimator != 'log'))) %>% 
  group_by(method, estimator, label) %>% 
  summarize(med_f1 = median(f1), std_f1 = sd(f1)) %>% 
  ungroup() %>% 
  arrange(label) %>% 
  ggplot(aes(x=factor(label), y=med_f1, fill=method)) + 
  geom_bar(stat="identity", position=position_dodge()) + 
  facet_wrap(~ estimator, labeller=labeller(estimator=estimator_names),
             scales = "free_x") + 
  scale_fill_manual(values=c('FC' = '#d95f02',
                             'KMC' = '#7570b3',
                             'CD' = '#4daf4a',
                             'SC' = '#ff7f00')) + 
  labs(x='Label', y=expression(Median~F[1]~Score), fill="Method",
       title=expression(F[1]~Score~vs~Label~by~Method~and~Classifier)) + 
  coord_flip() + 
  theme_bw()

ggsave('figures/fmow-performance-compare.pdf', plot=p, height=8, width=5)

# df %>% 
#   filter((method %in% c("SC", "KMC") & (estimator != 'log'))) %>% 
#   group_by(method, estimator, label) %>% 
#   summarize(med_f1 = median(f1)) %>% 
#   ungroup() %>% 
#   arrange(label) %>% 
#   ggplot(aes(x=factor(label), y=med_f1, fill=method)) +
#   geom_bar(stat='identity', position=position_dodge()) + 
#   facet_wrap(~estimator, labeller=labeller(estimator=estimator_names)) + 
#   scale_fill_manual(values=c('FC' = '#d95f02',
#                              'KMC' = '#7570b3',
#                              'CD' = '#4daf4a',
#                              'SC' = '#ff7f00')) + 
#   labs(x='Label', y=expression(Median~F[1]~Score), fill="Method") + 
#   coord_flip() + 
#   theme_bw()
# 
entropy = read_csv('leaf_stability_res.csv')
entropy = entropy %>%
  filter(label != -1) %>%
  inner_join(df, by=c('id', 'label')) %>%
  spread(key=metric, value=value) %>%
  group_by(method, estimator, label) %>%
  summarize(med_entropy = median(med_entropy),
            med_prob_conf = median(prob_conf)) %>%
  ungroup()

p = entropy %>% 
  filter(method %in% c("FC", "KMC") & (estimator == 'rf')) %>% 
  ggplot(aes(x=factor(label), y=med_prob_conf, fill=method)) + 
  geom_bar(stat='identity', position=position_dodge()) + 
  scale_fill_manual(values=c('FC' = '#d95f02',
                             'KMC' = '#7570b3',
                             'CD' = '#4daf4a',
                             'SC' = '#ff7f00')) + 
  labs(x='Label', y='Median Label Posterior Probability', fill='Method',
       title='Label Posterior Probability by Method') + 
  coord_flip() + 
  theme_bw()

ggsave('figures/fmow-label-posterior.pdf', plot=p, height=8, width=5)
