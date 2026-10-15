if (.Platform$OS.type == "windows" && !l10n_info()[["UTF-8"]])
  Sys.setlocale("LC_CTYPE", "English_United States.utf8")
source("R/common.R", encoding="UTF-8")
source("R/preprocessing.R", encoding="UTF-8")
RNGkind("Mersenne-Twister", "Inversion", "Rejection")
B <- protocol$bootstrap_replicates
required <- c("results/full/json/best_k5.json", "results/bootstrap/full_data_membership.csv",
  sprintf("results/bootstrap_models/bootstrap_%03d/best_k5.json",seq_len(B)),
  sprintf("results/bootstrap/bootstrap%d_membership.csv",seq_len(B)))
if (any(!file.exists(required))) stop("Run all required bootstrap fits first; missing: ",
  paste(required[!file.exists(required)],collapse=", "))
source("R/proposed_stability.R", encoding="UTF-8")
source("R/baseline_stability.R", encoding="UTF-8")
source("R/compare_stability.R", encoding="UTF-8")
save_environment("results/environment/R-stability-sessionInfo.txt")
