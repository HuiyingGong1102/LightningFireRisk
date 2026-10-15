if (.Platform$OS.type == "windows" && !l10n_info()[["UTF-8"]])
  Sys.setlocale("LC_CTYPE", "English_United States.utf8")
source("R/common.R", encoding="UTF-8")
source("R/preprocessing.R", encoding="UTF-8")
source("R/power_fit.R", encoding="UTF-8")
args <- commandArgs(trailingOnly=TRUE)
threads <- if(length(args)) as.integer(args[1]) else 1L
df <- read_events("data/raw/new_data.csv")
scaled <- transform_fire_data(df,readRDS("results/full/scaler.rds"))
dir.create("results/descriptive",recursive=TRUE,showWarnings=FALSE)
fits <- list()
for (block in c("temperature","precipitation")) {
  features <- if(block=="temperature") temp_features else rain_features
  Y <- t(as.matrix(scaled[paste0(features,"_index")]))
  index <- rowSums(Y)
  ord <- order(index)
  fits[[block]] <- power_equation_fit2(index[ord],Y[ord,,drop=FALSE],
    seq(min(index),max(index),length.out=50),thread=threads)
}
r2 <- function(y,fit) {
  sst <- sum((y-mean(y))^2)
  if(sst==0) return(NA_real_)
  1-sum((y-fit)^2)/sst
}
table <- data.frame(source_row_id=df$source_row_id)
for (block in names(fits)) {
  fit <- fits[[block]]
  table[[paste0(block,"_R2")]] <- vapply(seq_len(nrow(df)),function(i)
    r2(fit$original_data[,i],fit$power_fit[,i]),numeric(1))
}
write.csv(table,"results/descriptive/allometric_R2.csv",row.names=FALSE)
correlation <- cor(t(fits$temperature$original_data),t(fits$precipitation$original_data),method="spearman")
write.csv(correlation,"results/descriptive/temperature_precipitation_spearman.csv")
model <- jsonlite::fromJSON("results/full/json/best_k5.json")
module <- model$max_omega_logi+1L
if(length(module)!=nrow(df)) stop("Model/data length mismatch")
if("drought_index" %in% names(df)) {
  test <- kruskal.test(as.numeric(df$drought_index)~as.factor(module))
  writeLines(capture.output(test),"results/descriptive/drought_kruskal_wallis.txt")
}
write.csv(as.data.frame(table(module)),"results/descriptive/regime_sizes.csv",row.names=FALSE)
date <- as.Date(df$time)
write.csv(as.data.frame(table(module,month=as.integer(format(date,"%m")))),"results/descriptive/regime_month_counts.csv",row.names=FALSE)
set.seed(protocol$seed)
ids <- sample(seq_len(nrow(df)),min(5L,nrow(df)))
write.csv(data.frame(source_row_id=ids),"results/descriptive/representative_events.csv",row.names=FALSE)
pdf("results/descriptive/representative_trajectories.pdf",width=10,height=4)
par(mfrow=c(2,length(ids)),mar=c(3,3,2,1))
for(block in names(fits)) for(i in ids) {
  fit <- fits[[block]]
  plot(fit$Time,fit$original_data[,i],xlab="Environmental index",ylab=block,main=paste("Event",i))
  lines(fit$Time,fit$power_fit[,i],col="blue")
}
dev.off()
save_environment("results/environment/R-describe-sessionInfo.txt")
