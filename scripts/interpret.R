if (.Platform$OS.type == "windows" && !l10n_info()[["UTF-8"]])
  Sys.setlocale("LC_CTYPE", "English_United States.utf8")
source("R/common.R",encoding="UTF-8")
library(xgboost)
library(jsonlite)
RNGkind("Mersenne-Twister","Inversion","Rejection")
set.seed(protocol$seed)
df <- read_events("data/raw/new_data.csv")
features <- c(temp_features,rain_features,"drought_index")
if (!all(features %in% names(df))) stop("SHAP requires the drought-index column")
X <- as.matrix(data.frame(lapply(df[features],as.numeric),check.names=FALSE))
if(any(!is.finite(X))) stop("Invalid predictor values")
labels <- fromJSON("results/full/json/best_k5.json")$max_omega_logi
if(length(labels)!=nrow(X)) stop("Model/data length mismatch")
if(!identical(sort(unique(labels)),0:4)) stop("Interpretation requires all five regimes")
train_id <- sample(seq_len(nrow(X)),size=floor(protocol$xgboost$train_fraction*nrow(X)))
train <- xgb.DMatrix(X[train_id,,drop=FALSE],label=labels[train_id])
test <- xgb.DMatrix(X[-train_id,,drop=FALSE],label=labels[-train_id])
xgb_settings <- protocol$xgboost
fit <- xgb.train(params=list(objective="multi:softprob",num_class=protocol$selected_regimes,eval_metric="mlogloss",
  max_depth=xgb_settings$max_depth,eta=xgb_settings$eta,subsample=xgb_settings$subsample,
  colsample_bytree=xgb_settings$colsample_bytree,seed=protocol$seed,nthread=1),
  data=train,nrounds=xgb_settings$nrounds,verbose=0)
prob <- predict(fit,test,strict_shape=TRUE)
if(!identical(dim(prob),c(nrow(X)-length(train_id),5L))) stop("Unexpected prediction dimensions")
pred <- max.col(prob,ties.method="first")-1L
confusion <- table(factor(labels[-train_id],levels=0:4),factor(pred,levels=0:4))
tp <- diag(confusion)
precision <- tp/colSums(confusion)
recall <- tp/rowSums(confusion)
f1 <- 2*precision*recall/(precision+recall)
accuracy <- sum(tp)/sum(confusion)
chance <- sum(rowSums(confusion)*colSums(confusion))/sum(confusion)^2
out <- "results/interpretation"
dir.create(out,recursive=TRUE,showWarnings=FALSE)
write.csv(as.data.frame.matrix(confusion),file.path(out,"confusion_matrix.csv"))
write.csv(data.frame(module=1:5,precision=precision,recall=recall,F1=f1),file.path(out,"class_metrics.csv"),row.names=FALSE)
write.csv(data.frame(accuracy=accuracy,kappa=(accuracy-chance)/(1-chance)),file.path(out,"overall_metrics.csv"),row.names=FALSE)
write.csv(data.frame(source_row_id=df$source_row_id,subset=ifelse(seq_len(nrow(df)) %in% train_id,"train","test")),file.path(out,"split_membership.csv"),row.names=FALSE)
# strict_shape: event x class x (features + bias); drop the final bias column.
contrib <- predict(fit,train,predcontrib=TRUE,strict_shape=TRUE)
if(!identical(dim(contrib),c(length(train_id),5L,length(features)+1L))) stop("Unexpected SHAP dimensions")
importance <- apply(abs(contrib[,,seq_along(features),drop=FALSE]),3,mean)
result <- data.frame(feature=features,mean_absolute_SHAP=importance)
result <- result[order(result$mean_absolute_SHAP,decreasing=TRUE),]
write.csv(result,file.path(out,"SHAP_importance.csv"),row.names=FALSE,fileEncoding="UTF-8")
xgb.save(fit,file.path(out,"classifier.ubj"))
save_environment("results/environment/R-interpret-sessionInfo.txt")
