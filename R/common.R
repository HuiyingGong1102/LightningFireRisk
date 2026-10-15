if (dir.exists("renv/library")) .libPaths(c(normalizePath("renv/library"), .libPaths()))
protocol <- jsonlite::fromJSON("config/paper.json")
temp_features <- protocol$temperature_features
rain_features <- protocol$precipitation_features
drop_export_index <- function(data) {
  # R write.csv exports a sequential row-number column; never drop a real feature.
  if (names(data)[1] %in% c("", "X", "Unnamed: 0") &&
      identical(as.integer(data[[1]]), seq_len(nrow(data)))) data <- data[, -1, drop = FALSE]
  data
}
read_events <- function(path) {
  data <- drop_export_index(read.csv(path, check.names = FALSE, fileEncoding = "UTF-8"))
  data <- normalize_event_schema(data)
  required <- c(temp_features, rain_features, "time", "province")
  if (length(setdiff(required, names(data)))) stop("Missing columns: ", paste(setdiff(required, names(data)), collapse=", "))
  for (f in c(temp_features, rain_features)) {
    data[[f]] <- as.numeric(data[[f]])
    if (any(!is.finite(data[[f]]))) stop("Missing/non-finite values in ", f)
  }
  if (any(as.matrix(data[rain_features]) < 0)) stop("Negative precipitation")
  data$source_row_id <- seq_len(nrow(data))
  data
}
normalize_event_schema <- function(data) {
  # Unicode escapes retain compatibility with the original table headers.
  legacy_names <- c(
    "\u5e73\u5747\u6c14\u6e29", "\u6700\u9ad8\u6c14\u6e29", "\u6700\u4f4e\u6c14\u6e29",
    "\u8fd1\u5341\u65e5\u5e73\u5747\u6c14\u6e29", "\u8fd1\u4e09\u5341\u65e5\u5e73\u5747\u6c14\u6e29",
    "\u964d\u6c34", "\u8fc7\u53bb24\u5c0f\u65f6\u964d\u6c34\u5b9e\u51b5",
    "\u8fc7\u53bb\u4e09\u5929\u964d\u6c34\u5b9e\u51b5", "\u8fc7\u53bb\u5341\u5929\u964d\u6c34\u5b9e\u51b5",
    "\u8fc7\u53bb\u4e09\u5341\u5929\u964d\u6c34\u5b9e\u51b5",
    "\u65f6\u95f4", "\u7701\u4efd", "\u5e72\u65f1\u6307\u6570", "provience")
  canonical_names <- c(temp_features, rain_features, "time", "province", "drought_index", "province")
  for (i in seq_along(legacy_names)) {
    if (legacy_names[i] %in% names(data)) {
      if (canonical_names[i] %in% names(data))
        stop("Ambiguous duplicate column for ", canonical_names[i])
      names(data)[names(data) == legacy_names[i]] <- canonical_names[i]
    }
  }
  if ("province" %in% names(data)) {
    province <- trimws(as.character(data$province))
    province <- sub("\u7701$|\u81ea\u6cbb\u533a$", "", province)
    province[province == "\u5185\u8499\u53e4"] <- "Inner Mongolia"
    province[province == "\u9ed1\u9f99\u6c5f"] <- "Heilongjiang"
    province[province == "Inner Mongolia Autonomous Region"] <- "Inner Mongolia"
    province[province == "Heilongjiang Province"] <- "Heilongjiang"
    data$province <- province
  }
  data
}
save_environment <- function(path) {
  dir.create(dirname(path), recursive=TRUE, showWarnings=FALSE)
  writeLines(capture.output(sessionInfo()), path)
}
