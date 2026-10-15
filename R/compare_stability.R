# Compare all methods using the same bootstrap replicate IDs.
library(dplyr)
library(readr)
library(tidyr)

output_dir <- "results/stability_comparison"
dir.create(output_dir,recursive=TRUE,showWarnings=FALSE)
baseline <- read_csv("results/baseline_stability/bootstrap_stability_metrics.csv",show_col_types=FALSE) %>%
  select(bootstrap_id,method,ARI,NMI,Mean_Jaccard,Min_Jaccard)
proposed <- read_csv("results/proposed_stability/bootstrap_vs_reference_metrics.csv",show_col_types=FALSE)
jaccard <- read_csv("results/proposed_stability/bootstrap_vs_reference_module_jaccard.csv",show_col_types=FALSE) %>%
  group_by(bootstrap_id) %>% summarise(Mean_Jaccard=mean(Jaccard,na.rm=TRUE),
    Min_Jaccard=min(Jaccard,na.rm=TRUE),.groups="drop")
proposed <- proposed %>% left_join(jaccard,by="bootstrap_id") %>%
  mutate(method="Proposed") %>% select(bootstrap_id,method,ARI,NMI,Mean_Jaccard,Min_Jaccard)
metrics <- bind_rows(baseline,proposed)
if(anyNA(metrics[c("ARI","NMI","Mean_Jaccard","Min_Jaccard")])) stop("Incomplete bootstrap comparison")
summary <- metrics %>% pivot_longer(-c(bootstrap_id,method),names_to="metric",values_to="value") %>%
  group_by(method,metric) %>% summarise(n=n(),mean=mean(value),sd=sd(value),median=median(value),
    CI95_lower=quantile(value,.025),CI95_upper=quantile(value,.975),.groups="drop")
write_csv(metrics,file.path(output_dir,"all_method_bootstrap_metrics.csv"))
write_csv(summary,file.path(output_dir,"all_method_stability_summary_CI95.csv"))
