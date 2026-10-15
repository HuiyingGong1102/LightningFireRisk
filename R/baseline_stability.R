# ============================================================
# Bootstrap clustering stability comparison
# Methods:
#   1. K-means
#   2. GMM-diagonal (mclust VVI)
#   3. GMM-full     (mclust VVV)
#   4. Agglomerative clustering (Ward.D2)
#
# Metrics:
#   ARI, NMI, module-wise Jaccard, Mean Jaccard
#
# Important:
# - Each method is compared with its own full-data K=5 reference clustering.
# - ARI/NMI are label-permutation invariant.
# - Jaccard is calculated after Hungarian label alignment.
# - If existing bootstrap membership files are available, set
#   USE_EXISTING_MEMBERSHIP <- TRUE to ensure exactly the same
#   bootstrap samples as the proposed model.
# ============================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(purrr)
  library(tibble)
  library(mclust)
  library(clue)
})

set.seed(protocol$seed)

K <- protocol$selected_regimes
B <- protocol$bootstrap_replicates
KMEANS_NSTART <- protocol$initializations
KMEANS_ITER_MAX <- 1000L
REFIT_SCALER_EACH_BOOTSTRAP <- TRUE
USE_EXISTING_MEMBERSHIP <- TRUE

DATA_PATH <- "data/raw/new_data.csv"
MEMBERSHIP_DIR <- "results/bootstrap"
OUTPUT_DIR <- "results/baseline_stability"

dir.create(
  OUTPUT_DIR,
  recursive = TRUE,
  showWarnings = FALSE
)

df_clean <- read_events(DATA_PATH)

df_clean <- drop_export_index(df_clean)
df_source <- df_clean %>%
  mutate(source_row_id = row_number())

n_original <- nrow(df_source)

temp_features <- c(
  "Mean Temperature",
  "Maximum Temperature",
  "Minimum Temperature",
  "Mean Temperature over the Past 10 Days",
  "Mean Temperature over the Past 30 Days"
)

rain_features <- c(
  "Precipitation",
  "Observed Precipitation over the Past 24 Hours",
  "Observed Precipitation over the Past 3 Days",
  "Observed Precipitation over the Past 10 Days",
  "Observed Precipitation over the Past 30 Days"
)

model_cols <- c(
  paste0(temp_features, "_index"),
  paste0(rain_features, "_index")
)

required_functions <- c(
  "fit_fire_scaler",
  "transform_fire_data"
)

missing_functions <- required_functions[
  !vapply(
    required_functions,
    exists,
    logical(1),
    mode = "function"
  )
]

if (length(missing_functions) > 0) {
  stop(
    "Please define or source these functions first: ",
    paste(missing_functions, collapse = ", ")
  )
}

full_scaler <- fit_fire_scaler(
  train_data = df_source,
  temp_features = temp_features,
  rain_features = rain_features,
  lower_q = protocol$precipitation_quantiles[1],
  upper_q = protocol$precipitation_quantiles[2]
)

make_feature_matrix <- function(
  raw_data,
  scaler_to_use
) {
  scaled_data <- transform_fire_data(
    data = raw_data,
    scaler = scaler_to_use,
    clip_temperature = FALSE
  )

  missing_cols <- setdiff(
    model_cols,
    colnames(scaled_data)
  )

  if (length(missing_cols) > 0) {
    stop(
      "Missing transformed columns: ",
      paste(missing_cols, collapse = ", ")
    )
  }

  X <- as.matrix(
    scaled_data[, model_cols, drop = FALSE]
  )

  storage.mode(X) <- "double"

  if (any(!is.finite(X))) {
    stop("Feature matrix contains NA, NaN, or Inf.")
  }

  if (any(X <= 0)) {
    stop(
      "Feature matrix contains values <= 0; log2 cannot be calculated."
    )
  }

  X <- log2(X)
  rownames(X) <- raw_data$source_row_id

  X
}

X_full <- make_feature_matrix(
  raw_data = df_source,
  scaler_to_use = full_scaler
)

fit_kmeans_labels <- function(
  X,
  seed
) {
  set.seed(seed)

  fit <- stats::kmeans(
    x = X,
    centers = K,
    nstart = KMEANS_NSTART,
    iter.max = KMEANS_ITER_MAX,
    algorithm = "Hartigan-Wong"
  )

  as.integer(fit$cluster)
}

fit_mclust_labels <- function(
    X,
    model_name,
    seed
) {

  set.seed(seed)

  X <- as.matrix(X)
  storage.mode(X) <- "double"

  if (any(!is.finite(X))) {
    stop(
      model_name,
      " input contains NA, NaN or Inf."
    )
  }

  if (nrow(X) <= ncol(X)) {
    stop(
      model_name,
      " requires more observations than variables."
    )
  }

  # ========================================================
  # First attempt: standard Mclust fitting
  # ========================================================
  fit <- tryCatch(
    {
      mclust::Mclust(
        data = X,
        G = K,
        modelNames = model_name,
        verbose = FALSE,
        warn = TRUE
      )
    },
    warning = function(w) {

      message(
        model_name,
        " initial fit warning: ",
        conditionMessage(w)
      )

      # Return NULL to retry with regularization rather than stop here.
      NULL
    },
    error = function(e) {

      message(
        model_name,
        " initial fit failed: ",
        conditionMessage(e)
      )

      NULL
    }
  )

  fit_is_valid <- function(fit_object) {

    if (is.null(fit_object)) {
      return(FALSE)
    }

    labels <- fit_object$classification

    if (is.null(labels)) {
      return(FALSE)
    }

    if (length(labels) != nrow(X)) {
      return(FALSE)
    }

    if (anyNA(labels)) {
      return(FALSE)
    }

    if (length(unique(labels)) != K) {
      return(FALSE)
    }

    TRUE
  }

  # ========================================================
  # Second attempt: regularization with a conjugate prior
  # ========================================================
  if (!fit_is_valid(fit)) {

    set.seed(seed + 10000L)

    fit <- tryCatch(
      {
        mclust::Mclust(
          data = X,
          G = K,
          modelNames = model_name,

          prior = mclust::priorControl(
            functionName = "defaultPrior",
            shrinkage = 0.01
          ),

          verbose = FALSE,
          warn = TRUE
        )
      },
      warning = function(w) {

        message(
          model_name,
          " regularized fit warning: ",
          conditionMessage(w)
        )

        NULL
      },
      error = function(e) {

        message(
          model_name,
          " regularized fit failed: ",
          conditionMessage(e)
        )

        NULL
      }
    )
  }

  # ========================================================
  # Third attempt: small jitter plus regularization
  # Handle duplicate observations and numerical singularities.
  # ========================================================
  if (!fit_is_valid(fit)) {

    set.seed(seed + 20000L)

    jitter_scale <- 1e-7

    X_jitter <- X + matrix(
      rnorm(
        length(X),
        mean = 0,
        sd = jitter_scale
      ),
      nrow = nrow(X),
      ncol = ncol(X)
    )

    fit <- tryCatch(
      {
        mclust::Mclust(
          data = X_jitter,
          G = K,
          modelNames = model_name,

          prior = mclust::priorControl(
            functionName = "defaultPrior",
            shrinkage = 0.01
          ),

          verbose = FALSE,
          warn = TRUE
        )
      },
      warning = function(w) {

        message(
          model_name,
          " jittered fit warning: ",
          conditionMessage(w)
        )

        NULL
      },
      error = function(e) {

        message(
          model_name,
          " jittered fit failed: ",
          conditionMessage(e)
        )

        NULL
      }
    )
  }

  # ========================================================
  # Final validation
  # ========================================================
  if (!fit_is_valid(fit)) {

    stop(
      model_name,
      " could not obtain a valid K=",
      K,
      " clustering. Possible causes include singular covariance, ",
      "degenerate components or too few observations in a regime."
    )
  }

  labels <- as.integer(
    fit$classification
  )

  if (length(labels) != nrow(X)) {
    stop(
      model_name,
      " label count mismatch: expected ",
      nrow(X),
      ", got ",
      length(labels)
    )
  }

  labels
}

fit_agglomerative_labels <- function(
  X
) {
  distance_matrix <- stats::dist(
    X,
    method = "euclidean"
  )

  hierarchical_fit <- stats::hclust(
    distance_matrix,
    method = "ward.D2"
  )

  as.integer(
    stats::cutree(
      hierarchical_fit,
      k = K
    )
  )
}

fit_all_methods <- function(
  X,
  seed
) {
  list(
    Kmeans = fit_kmeans_labels(
      X = X,
      seed = seed + 1L
    ),

    GMM_diagonal = fit_mclust_labels(
      X = X,
      model_name = "VVI",
      seed = seed + 2L
    ),

    GMM_full = fit_mclust_labels(
      X = X,
      model_name = "VVV",
      seed = seed + 3L
    ),

    Agglomerative = fit_agglomerative_labels(
      X = X
    )
  )
}

calculate_nmi <- function(
  labels1,
  labels2
) {
  valid <- complete.cases(
    labels1,
    labels2
  )

  labels1 <- labels1[valid]
  labels2 <- labels2[valid]

  contingency <- table(
    labels1,
    labels2
  )

  total_n <- sum(contingency)

  if (total_n == 0) {
    return(NA_real_)
  }

  joint_probability <- contingency / total_n
  row_probability <- rowSums(joint_probability)
  col_probability <- colSums(joint_probability)

  nonzero <- which(
    joint_probability > 0,
    arr.ind = TRUE
  )

  mutual_information <- sum(
    joint_probability[nonzero] *
      log(
        joint_probability[nonzero] /
          (
            row_probability[nonzero[, 1]] *
              col_probability[nonzero[, 2]]
          )
      )
  )

  entropy1 <- -sum(
    row_probability[row_probability > 0] *
      log(row_probability[row_probability > 0])
  )

  entropy2 <- -sum(
    col_probability[col_probability > 0] *
      log(col_probability[col_probability > 0])
  )

  if (entropy1 <= 0 || entropy2 <= 0) {
    return(NA_real_)
  }

  mutual_information /
    sqrt(entropy1 * entropy2)
}

collapse_duplicate_labels <- function(
  source_ids,
  labels
) {
  if (length(source_ids) != length(labels)) {
    stop(
      "source_ids and labels have different lengths."
    )
  }

  tibble(
    source_row_id = as.integer(source_ids),
    bootstrap_label = as.integer(labels)
  ) %>%
    group_by(source_row_id) %>%
    summarise(
      bootstrap_label = {
        label_table <- table(bootstrap_label)

        tied_winners <- as.integer(
          names(
            label_table[
              label_table == max(label_table)
            ]
          )
        )

        min(tied_winners)
      },

      bootstrap_copies = n(),

      number_of_distinct_labels =
        n_distinct(bootstrap_label),

      .groups = "drop"
    )
}

align_to_reference <- function(
  bootstrap_labels,
  reference_labels
) {
  contingency <- table(
    factor(
      bootstrap_labels,
      levels = seq_len(K)
    ),
    factor(
      reference_labels,
      levels = seq_len(K)
    )
  )

  assignment <- clue::solve_LSAP(
    contingency,
    maximum = TRUE
  )

  mapping <- as.integer(assignment)
  aligned_labels <- mapping[bootstrap_labels]

  list(
    aligned_labels = aligned_labels,
    mapping = mapping,
    contingency = contingency
  )
}

calculate_module_jaccard <- function(
  aligned_bootstrap_labels,
  reference_labels,
  module_id
) {
  bootstrap_set <- which(
    aligned_bootstrap_labels == module_id
  )

  reference_set <- which(
    reference_labels == module_id
  )

  union_set <- union(
    bootstrap_set,
    reference_set
  )

  intersection_set <- intersect(
    bootstrap_set,
    reference_set
  )

  if (length(union_set) == 0) {
    return(NA_real_)
  }

  length(intersection_set) /
    length(union_set)
}

calculate_stability_metrics <- function(
  bootstrap_assignment,
  reference_assignment,
  method_name,
  bootstrap_id
) {
  common_data <- bootstrap_assignment %>%
    inner_join(
      reference_assignment,
      by = "source_row_id"
    )

  if (nrow(common_data) < K) {
    stop(
      "Too few common samples for bootstrap ",
      bootstrap_id,
      ", method ",
      method_name
    )
  }

  alignment <- align_to_reference(
    bootstrap_labels =
      common_data$bootstrap_label,

    reference_labels =
      common_data$reference_label
  )

  common_data$aligned_bootstrap_label <-
    alignment$aligned_labels

  ari_value <- mclust::adjustedRandIndex(
    common_data$aligned_bootstrap_label,
    common_data$reference_label
  )

  nmi_value <- calculate_nmi(
    common_data$aligned_bootstrap_label,
    common_data$reference_label
  )

  module_jaccard <- map_dfr(
    seq_len(K),
    function(module_id) {
      tibble(
        bootstrap_id = bootstrap_id,
        method = method_name,
        module = module_id,

        Jaccard = calculate_module_jaccard(
          aligned_bootstrap_labels =
            common_data$aligned_bootstrap_label,

          reference_labels =
            common_data$reference_label,

          module_id = module_id
        )
      )
    }
  )

  valid_jaccard <- module_jaccard$Jaccard[
    is.finite(module_jaccard$Jaccard)
  ]

  metric_row <- tibble(
    bootstrap_id = bootstrap_id,
    method = method_name,
    n_bootstrap_rows =
      sum(bootstrap_assignment$bootstrap_copies),
    n_unique_bootstrap_samples =
      nrow(bootstrap_assignment),
    n_common_samples =
      nrow(common_data),
    ARI = ari_value,
    NMI = nmi_value,
    Mean_Jaccard = if (
      length(valid_jaccard) > 0
    ) {
      mean(valid_jaccard)
    } else {
      NA_real_
    },
    Min_Jaccard = if (
      length(valid_jaccard) > 0
    ) {
      min(valid_jaccard)
    } else {
      NA_real_
    },
    inconsistent_duplicate_samples =
      sum(
        bootstrap_assignment$
          number_of_distinct_labels > 1
      )
  )

  mapping_row <- tibble(
    bootstrap_id = bootstrap_id,
    method = method_name,
    bootstrap_raw_module = seq_len(K),
    aligned_reference_module =
      alignment$mapping
  )

  list(
    metrics = metric_row,
    module_jaccard = module_jaccard,
    mapping = mapping_row
  )
}

cat("Fitting full-data reference clusterings...\n")

reference_labels_list <- fit_all_methods(
  X = X_full,
  seed = protocol$seed
)

reference_assignments <- imap(
  reference_labels_list,
  function(labels, method_name) {
    tibble(
      source_row_id = df_source$source_row_id,
      reference_label = as.integer(labels)
    )
  }
)

reference_label_table <- imap_dfr(
  reference_labels_list,
  function(labels, method_name) {
    tibble(
      source_row_id = df_source$source_row_id,
      method = method_name,
      reference_label = as.integer(labels)
    )
  }
)

write_csv(
  reference_label_table,
  file.path(
    OUTPUT_DIR,
    "full_data_reference_labels.csv"
  )
)

reference_cluster_sizes <- reference_label_table %>%
  dplyr::group_by(
    method,
    reference_label
  ) %>%
  dplyr::summarise(
    n = dplyr::n(),
    .groups = "drop_last"
  ) %>%
  dplyr::mutate(
    proportion = n / sum(n)
  ) %>%
  dplyr::ungroup()

write_csv(
  reference_cluster_sizes,
  file.path(
    OUTPUT_DIR,
    "full_data_reference_cluster_sizes.csv"
  )
)

if (!USE_EXISTING_MEMBERSHIP) {
  set.seed(protocol$seed)

  bootstrap_indices <- lapply(
    seq_len(B),
    function(b) {
      sample.int(
        n = n_original,
        size = n_original,
        replace = TRUE
      )
    }
  )
}

get_bootstrap_indices <- function(
  bootstrap_id
) {
  if (USE_EXISTING_MEMBERSHIP) {
    membership_path <- file.path(
      MEMBERSHIP_DIR,
      paste0(
        "bootstrap",
        bootstrap_id,
        "_membership.csv"
      )
    )

    if (!file.exists(membership_path)) {
      stop(
        "Membership file does not exist: ",
        membership_path
      )
    }

    membership <- read_csv(
      membership_path,
      show_col_types = FALSE
    )

    if (!"source_row_id" %in% names(membership)) {
      stop(
        "source_row_id is absent from: ",
        membership_path
      )
    }

    indices <- as.integer(
      membership$source_row_id
    )
  } else {
    indices <- bootstrap_indices[[bootstrap_id]]
  }

  if (
    length(indices) != n_original ||
    any(indices < 1L) ||
    any(indices > n_original)
  ) {
    stop(
      "Invalid bootstrap indices for bootstrap ",
      bootstrap_id
    )
  }

  indices
}

metric_results <- vector(
  mode = "list",
  length = B
)

jaccard_results <- vector(
  mode = "list",
  length = B
)

mapping_results <- vector(
  mode = "list",
  length = B
)

failure_results <- list()

for (b in seq_len(B)) {
  cat(
    sprintf(
      "\rProcessing bootstrap %03d/%03d",
      b,
      B
    )
  )

  current_failure <- list()
  current_metrics <- list()
  current_jaccard <- list()
  current_mapping <- list()

  bootstrap_ids <- get_bootstrap_indices(
    bootstrap_id = b
  )

  bootstrap_raw <- df_source[
    bootstrap_ids,
    ,
    drop = FALSE
  ]

  if (REFIT_SCALER_EACH_BOOTSTRAP) {
    bootstrap_scaler <- fit_fire_scaler(
      train_data = bootstrap_raw,
      temp_features = temp_features,
      rain_features = rain_features,
      lower_q = protocol$precipitation_quantiles[1],
      upper_q = protocol$precipitation_quantiles[2]
    )
  } else {
    bootstrap_scaler <- full_scaler
  }

  X_bootstrap <- make_feature_matrix(
    raw_data = bootstrap_raw,
    scaler_to_use = bootstrap_scaler
  )

  method_labels <- tryCatch(
    {
      fit_all_methods(
        X = X_bootstrap,
        seed = protocol$seed + b * 100L
      )
    },
    error = function(e) {
      current_failure[["all_methods"]] <<- tibble(
        bootstrap_id = b,
        method = "all_methods",
        error_message = conditionMessage(e)
      )
      NULL
    }
  )

  if (is.null(method_labels)) {
    failure_results[[length(failure_results) + 1L]] <-
      bind_rows(current_failure)
    next
  }

  for (
    method_name in names(method_labels)
  ) {
    tryCatch(
      {
        collapsed_assignment <-
          collapse_duplicate_labels(
            source_ids = bootstrap_ids,
            labels = method_labels[[method_name]]
          )

        stability_result <-
          calculate_stability_metrics(
            bootstrap_assignment =
              collapsed_assignment,

            reference_assignment =
              reference_assignments[[method_name]],

            method_name = method_name,
            bootstrap_id = b
          )

        current_metrics[[method_name]] <- stability_result$metrics

        current_jaccard[[method_name]] <- stability_result$module_jaccard

        current_mapping[[method_name]] <- stability_result$mapping
      },
      error = function(e) {
        current_failure[[method_name]] <<- tibble(
          bootstrap_id = b,
          method = method_name,
          error_message = conditionMessage(e)
        )
      }
    )
  }

  metric_results[[b]] <-
    bind_rows(current_metrics)

  jaccard_results[[b]] <-
    bind_rows(current_jaccard)

  mapping_results[[b]] <-
    bind_rows(current_mapping)

  if (length(current_failure) > 0) {
    failure_results[[length(failure_results) + 1L]] <- bind_rows(current_failure)
  }

  write_csv(
    bind_rows(
      metric_results[
        seq_len(b)
      ]
    ),
    file.path(
      OUTPUT_DIR,
      "bootstrap_stability_metrics_partial.csv"
    )
  )
}

cat("\nBootstrap analysis completed.\n")

stability_metrics <- bind_rows(
  metric_results
)

module_jaccard_results <- bind_rows(
  jaccard_results
)

label_mapping_results <- bind_rows(
  mapping_results
)

failure_table <- bind_rows(
  failure_results
)

write_csv(
  stability_metrics,
  file.path(
    OUTPUT_DIR,
    "bootstrap_stability_metrics.csv"
  )
)

write_csv(
  module_jaccard_results,
  file.path(
    OUTPUT_DIR,
    "bootstrap_module_jaccard.csv"
  )
)

write_csv(
  label_mapping_results,
  file.path(
    OUTPUT_DIR,
    "bootstrap_label_mapping.csv"
  )
)

write_csv(
  failure_table,
  file.path(
    OUTPUT_DIR,
    "bootstrap_failures.csv"
  )
)

summarise_metric <- function(
  x
) {
  x <- x[
    is.finite(x)
  ]

  if (length(x) == 0) {
    return(
      tibble(
        n = 0L,
        mean = NA_real_,
        sd = NA_real_,
        median = NA_real_,
        CI95_lower = NA_real_,
        CI95_upper = NA_real_
      )
    )
  }

  tibble(
    n = length(x),
    mean = mean(x),
    sd = stats::sd(x),
    median = stats::median(x),
    CI95_lower = unname(
      stats::quantile(
        x,
        probs = 0.025,
        na.rm = TRUE
      )
    ),
    CI95_upper = unname(
      stats::quantile(
        x,
        probs = 0.975,
        na.rm = TRUE
      )
    )
  )
}

stability_summary <- stability_metrics %>%
  select(
    method,
    ARI,
    NMI,
    Mean_Jaccard,
    Min_Jaccard
  ) %>%
  pivot_longer(
    cols = -method,
    names_to = "metric",
    values_to = "value"
  ) %>%
  group_by(
    method,
    metric
  ) %>%
  group_modify(
    ~ summarise_metric(.x$value)
  ) %>%
  ungroup()

module_jaccard_summary <-
  module_jaccard_results %>%
  group_by(
    method,
    module
  ) %>%
  group_modify(
    ~ summarise_metric(.x$Jaccard)
  ) %>%
  ungroup()

write_csv(
  stability_summary,
  file.path(
    OUTPUT_DIR,
    "bootstrap_stability_summary_CI95.csv"
  )
)

write_csv(
  module_jaccard_summary,
  file.path(
    OUTPUT_DIR,
    "bootstrap_module_jaccard_summary_CI95.csv"
  )
)

publication_table <- stability_summary %>%
  mutate(
    estimate_CI = sprintf(
      "%.3f [%.3f, %.3f]",
      median,
      CI95_lower,
      CI95_upper
    )
  ) %>%
  select(
    method,
    metric,
    estimate_CI
  ) %>%
  pivot_wider(
    names_from = metric,
    values_from = estimate_CI
  )

write_csv(
  publication_table,
  file.path(
    OUTPUT_DIR,
    "bootstrap_stability_publication_table.csv"
  )
)

print(publication_table)
