library(tidyverse)

setwd('c:/devspace/mycode/label_reduction_data/fmow')
exp_df = read_csv('experiment_settings.csv')
group_df = read_csv('group_res.csv')
df = exp_df %>% inner_join(group_df, by='id')

# This issue occurs with the RF estimator
df = df %>% filter(estimator == 'rf')

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

df = df %>% 
  mutate(method = map2_chr(df$method, df$group_algo, adjust_method))

# Function to compare label groupings between the methods
get_grouping = function(df, method, run_num, is_cd, ...) {
  filter_args = list(...)
  df = df %>% 
    filter((method == !!method) & (run_num == !!run_num))
  
  if (is_cd == TRUE) {
    df = df %>%
      filter(metrics == filter_args$metrics)
  } else {
    df = df %>% 
      filter(k == filter_args$k)
  }
  
  # Remove duplicate values because it's possible that we multiple runs stored
  # in the data
  df = df %>% 
    distinct(label, .keep_all = T)
  score = df %>% pull(group) %>% table(.) %>% as.numeric %>% max
  return(score)
}

# Go through each of the unique value combinations and get the grouping score
get_score = function(df) {
  sc_kmc_grid = expand.grid(method=c('KMC', 'SC'), run_num=1:50, k=2:61)
  n = nrow(sc_kmc_grid)
  scores = rep(0, n)
  for (i in 1:n) {
    method = sc_kmc_grid[i, 1]
    run_num = sc_kmc_grid[i, 2]
    k = sc_kmc_grid[i, 3]
    scores[i] = get_grouping(df, method=method, run_num=run_num, is_cd=F,
                             k=k)
  }
  
  sc_kmc_grid$score = scores
  
  cd_grid = expand.grid(method='CD',
                        run_num=1:50,
                        metrics=c('comm-rbf', 'comm-l2', 'comm-linf',
                                  'comm-emd'))
  n = nrow(cd_grid)
  scores = rep(0, n)
  for (i in 1:n) {
    run_num = cd_grid[i, 2]
    metrics = cd_grid[i, 3]
    scores[i] = get_grouping(df, method='CD', run_num=run_num, is_cd=T,
                             metrics=metrics)
  }
  
  cd_grid$score = scores
  grid_res = as.tibble(sc_kmc_grid) %>% 
    bind_rows(cd_grid) %>% 
    mutate(metrics = as.character(metrics))
  return(grid_res)
}

score_df = get_score(df)

p = score_df %>% 
  filter(method != 'CD') %>% 
  group_by(method, k) %>% 
  summarize(med_score=median(score)) %>% 
  ggplot(aes(x=k, y=med_score, group=method)) + 
  geom_point(aes(shape=method, color=method), size=2) + 
  scale_color_manual(values=c('KMC' = '#377eb8', 'SC' = '#ff7f00'),
                     name='Method') +
  scale_shape_manual(values=c('KMC' = 'square', 'SC' = 'triangle'),
                     name='Method') + 
  labs(x='Number of Meta-Classes', y='Median Maximum Meta-Class Size',
       title='Meta-Class Imbalance by HC Method') + 
  theme_bw()

ggsave('figures/rf-metaclass-imbalance.pdf', plot=p, scale=0.7, height=5, 
       width=8)

p = score_df %>% 
  filter(method == 'CD') %>% 
  ggplot(aes(x=metrics, y=score)) + 
  geom_boxplot() + 
  labs(x='', y='Median Maximum Meta-Class Size', 
       title='Community Detection Meta-Class Imbalance') + 
  scale_x_discrete(labels=c('comm-emd' = 'EMD', 
                            'comm-l2' = expression(L[2]),
                            'comm-linf' = expression(L[infinity]), 
                            'comm-rbf' = 'RBF')) + 
  theme_bw() + 
  theme(axis.text.x = element_text(size=14))

ggsave('figures/cd-metaclass-imbalance.pdf', plot=p, scale=0.65, height=5,
       width=8.09)
