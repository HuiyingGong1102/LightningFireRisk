# Run from the repository root: Rscript --vanilla scripts/prepare.R [input]
if (.Platform$OS.type == "windows" && !l10n_info()[["UTF-8"]])
  Sys.setlocale("LC_CTYPE", "English_United States.utf8")
source("R/common.R", encoding="UTF-8")
source("R/preprocessing.R", encoding="UTF-8")
args <- commandArgs(trailingOnly=TRUE)
input <- if(length(args)) args[1] else "data/raw/new_data.csv"
B <- protocol$bootstrap_replicates
if (!is.finite(B) || B < 1) stop("B must be positive")
RNGkind("Mersenne-Twister", "Inversion", "Rejection")
df <- read_events(input)
dir.create("results", showWarnings=FALSE)
writeLines(as.character(tools::md5sum(input)), "results/input_md5.txt")

prepare_split <- function(train, test=NULL, directory) {
  dir.create(directory, recursive=TRUE, showWarnings=FALSE)
  scaler <- fit_fire_scaler(train, temp_features, rain_features,
    lower_q=protocol$precipitation_quantiles[1], upper_q=protocol$precipitation_quantiles[2])
  scaled <- transform_fire_data(train, scaler, clip_temperature=FALSE)
  test_scaled <- if(!is.null(test)) transform_fire_data(test, scaler, clip_temperature=FALSE) else NULL
  blocks <- list(temp_features, rain_features)
  matrices <- times <- test_matrices <- list()
  order_records <- list()
  for (b in 1:2) {
    cols <- paste0(blocks[[b]], "_index")
    values <- t(as.matrix(scaled[cols]))
    index <- rowSums(values)
    order <- order(index)
    matrices[[b]] <- log2(values[order,,drop=FALSE])
    times[[b]] <- log2(index[order])
    order_records[[b]] <- data.frame(block=b, position=1:5, feature=cols[order], index=index[order])
    if(!is.null(test)) test_matrices[[b]] <- log2(t(as.matrix(test_scaled[cols]))[order,,drop=FALSE])
  }
  X <- do.call(rbind,matrices)
  if(any(!is.finite(X))) stop("Nonpositive scaled temperature or non-finite training input: ",directory)
  colnames(X) <- paste0("sample_",seq_len(nrow(train)))
  write.csv(X,file.path(directory,"data_all.csv"),fileEncoding="UTF-8")
  time <- do.call(c,times)
  write.csv(t(time),file.path(directory,"time_all.csv"),fileEncoding="UTF-8")
  write.csv(train[c("source_row_id")],file.path(directory,"membership.csv"),row.names=FALSE)
  write.csv(do.call(rbind,order_records),file.path(directory,"feature_order.csv"),row.names=FALSE,fileEncoding="UTF-8")
  saveRDS(scaler,file.path(directory,"scaler.rds"))
  if(!is.null(test)) {
    TX <- do.call(rbind,test_matrices)
    if(any(!is.finite(TX))) stop("Nonpositive scaled temperature or non-finite test input: ",directory)
    colnames(TX) <- paste0("sample_",test$source_row_id)
    write.csv(TX,file.path(directory,"test_data_all.csv"),fileEncoding="UTF-8")
    write.csv(test[c("source_row_id")],file.path(directory,"test_membership.csv"),row.names=FALSE)
  }
  invisible(NULL)
}

# Manuscript Eq. 1: contributions are aggregated over training events.
prepare_split(df,directory="results/full")
set.seed(protocol$seed)
fold <- sample(rep(seq_len(protocol$cv_folds),length.out=nrow(df)))
write.csv(data.frame(source_row_id=df$source_row_id,fold=fold),"results/folds.csv",row.names=FALSE)
for(f in seq_len(protocol$cv_folds)) prepare_split(df[fold!=f,],df[fold==f,],paste0("results/cv/fold_",f))
years <- as.integer(format(as.Date(df$time),"%Y"))
if(anyNA(years)) stop("Invalid event dates")
limits <- protocol$temporal_splits
for(i in seq_len(nrow(limits))) {
  lim <- limits[i,]
  prepare_split(df[years>=2005 & years<=lim[1],],df[years>=lim[2] & years<=lim[3],],paste0("results/temporal/split_",i))
}
province <- df$province
if(any(!province %in% c("Inner Mongolia","Heilongjiang"))) stop("Unexpected province")
prepare_split(df[province=="Heilongjiang",],df[province=="Inner Mongolia",],"results/spatial/split_IM")
prepare_split(df[province=="Inner Mongolia",],df[province=="Heilongjiang",],"results/spatial/split_HLJ")
dir.create("results/bootstrap",showWarnings=FALSE)
prepare_split(df,directory="results/bootstrap/full_data")
write.csv(data.frame(source_row_id=df$source_row_id),"results/bootstrap/full_data_membership.csv",row.names=FALSE)
set.seed(protocol$seed)
for(b in seq_len(B)) {
  ids <- sample.int(nrow(df),nrow(df),replace=TRUE)
  directory <- sprintf("results/bootstrap/bootstrap_%03d",b)
  prepare_split(df[ids,,drop=FALSE],directory=directory)
  write.csv(data.frame(sample_name=paste0("sample_",seq_along(ids)),bootstrap_id=b,
    bootstrap_row_id=seq_along(ids),source_row_id=ids),
    sprintf("results/bootstrap/bootstrap%d_membership.csv",b),row.names=FALSE)
}
save_environment("results/environment/R-sessionInfo.txt")
cat("Prepared full, CV, temporal, spatial and",B,"bootstrap datasets.\n")
