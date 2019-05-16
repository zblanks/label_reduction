library(tidyverse)
library(gghighlight)
library(ggrepel)

setwd('c:/devspace/mycode/label_reduction_data/fmow')
exp_df = read_csv('experiment_settings.csv')
search_df = read_csv('search_res.csv')
df = exp_df %>% inner_join(search_df, by='id')

adjust_method = function(method, group_algo) {
  # Establish of series of if-then statements to generate the new method
  if (method == 'f') {
    return('FC')
  } else {
    # Adjust depending on the grouping algorithm
    if (group_algo == 'kmm') {
      return('KMC')
    } else if (group_algo == 'comm') {
      return('CD')
    } else if (group_algo == 'lp') {
      return('P')
    } else if (group_algo == 'kmm-sc') {
      return('KMC-SC')
    } else {
      return('SC')
    }
  }
}

adjust_estimator = function(estimator) {
  if (estimator == 'rf') {
    return('Random Forest')
  } else if (estimator == 'knn') {
    return('KNN')
  } else {
    return('Logistic Reg.')
  }
}

df = df %>% 
  mutate(method = map2_chr(df$method, df$group_algo, adjust_method))

df = df %>% 
  mutate(estimator = map_chr(df$estimator, adjust_estimator))

#### KMC Hyer-parameter Search ####

sub_df = df %>% 
  filter((method == 'KMC') & (metric == 'top1') & 
           (estimator != 'Logistic Reg.')) %>% 
  group_by(k, estimator) %>% 
  summarize(avg_val = mean(value)) %>% 
  ungroup()

knn_max = sub_df %>% filter(estimator == 'KNN') %>% pull(avg_val) %>% max
rf_max = sub_df %>% filter(estimator == 'Random Forest') %>% pull(avg_val) %>% max

set.seed(17)

p = sub_df %>% 
  ggplot(aes(x=k, y=avg_val, color=estimator)) +
  geom_point(size=2) + 
  gghighlight(avg_val %in% c(knn_max, rf_max), label_key=estimator,
              label_params=list(box.padding=1.25)) + 
  scale_color_manual(values=c('KNN' = '#7570b3', 'Random Forest' = '#d95f02')) + 
  labs(x='Number of Meta-Classes', y='Average Leaf Top 1',
       title='KMC Hyper-Parameter Search') + 
  theme_minimal()

ggsave('figures/kmc-hyperparameter.pdf', plot=p, height=5, width=8, 
       scale=0.7)

#### CD Hyper-parameter search ####

cd_df = df %>% 
  filter((method == 'CD') & (metric == 'top1') & (estimator != 'Logistic Reg.'))

p = cd_df %>% 
  ggplot(aes(x=metrics, y=value)) + 
  geom_boxplot() + 
  facet_wrap(~ estimator, scales = 'free') + 
  labs(x='', y='Leaf Top 1', 
       title='Community Detection Hyper-Parameter Search') + 
  scale_x_discrete(labels=c('comm-emd' = 'EMD', 
                            'comm-l2' = expression(L[2]),
                            'comm-linf' = expression(L[infinity]), 
                            'comm-rbf' = 'RBF')) + 
  theme_bw() + 
  theme(axis.text.x = element_text(size=14))

ggsave('figures/cd-hyperparameter.pdf', plot=p, height=5, width=8, scale=0.9)
