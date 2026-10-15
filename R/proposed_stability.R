
# ============================================================
# 0. Load R packages for stability analysis
# ============================================================
library(dplyr)
library(readr)
library(tidyr)
library(purrr)
library(jsonlite)
library(clue)
library(mclust)

# ============================================================
# 1. Paths and settings
# ============================================================
K <- protocol$selected_regimes
B <- protocol$bootstrap_replicates

BOOTSTRAP_INPUT_DIR <-
  "results/bootstrap"

BOOTSTRAP_RESULT_DIR <-
  "results/bootstrap_models"

REFERENCE_JSON <-
  "results/full/json/best_k5.json"

REFERENCE_MEMBERSHIP <-
  file.path(
    BOOTSTRAP_INPUT_DIR,
    "full_data_membership.csv"
  )

STABILITY_OUTPUT_DIR <-
  "results/proposed_stability"

dir.create(
  STABILITY_OUTPUT_DIR,
  showWarnings = FALSE,
  recursive = TRUE
)

# ============================================================
# 2. Read a model JSON file
# ============================================================
read_cluster_model <- function(
    json_path,
    K = 5L
) {

  if (!file.exists(json_path)) {
    stop(
      "Model file does not exist: ",
      json_path
    )
  }

  model <- jsonlite::fromJSON(
    json_path
  )

  # The Python schema always uses zero-based labels, even if module 0 is empty.
  labels <- as.integer(unlist(model$max_omega_logi)) + 1L

  if (any(!labels %in% seq_len(K))) {
    stop(
      "Model labels must be between 1 and ",
      K,
      ": ",
      json_path
    )
  }

  par <- as.numeric(
    unlist(model$par)
  )

  prob_log <- as.numeric(
    unlist(model$prob_log)
  )

  if (length(par) != 4 + 4 * K) {
    stop(
      "Parameter count mismatch: ",
      json_path,
      ", got ",
      length(par)
    )
  }

  if (length(prob_log) != K) {
    stop(
      "prob_log length mismatch: ",
      json_path
    )
  }

  # Convert log weights to prior probabilities stably.
  prior_probability <- exp(
    prob_log - max(prob_log)
  )

  prior_probability <-
    prior_probability /
    sum(prior_probability)

  list(
    labels = labels,
    par = par,
    prob_log = prob_log,
    prior_probability = prior_probability,
    raw_model = model
  )
}

# ============================================================
# 3. Aggregate labels for repeated bootstrap observations
# ============================================================
collapse_bootstrap_labels <- function(
    membership_df,
    labels
) {

  if (nrow(membership_df) != length(labels)) {
    stop(
      "Membership row count differs from the model label count: ",
      nrow(membership_df),
      " vs ",
      length(labels)
    )
  }

  tibble(
    source_row_id = as.character(
      membership_df$source_row_id
    ),
    raw_module = as.integer(labels)
  ) %>%
    group_by(source_row_id) %>%
    summarise(
      raw_module = {

        frequency_table <- table(
          raw_module
        )

        winner_labels <- as.integer(
          names(
            frequency_table[
              frequency_table ==
                max(frequency_table)
            ]
          )
        )

        min(winner_labels)
      },

      bootstrap_copies = n(),

      number_of_assigned_modules =
        n_distinct(raw_module),

      .groups = "drop"
    )
}

# ============================================================
# 4. Calculate normalized mutual information
# Normalize by sqrt(Hx * Hy).
# ============================================================
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

  n_total <- sum(
    contingency
  )

  if (n_total == 0) {
    return(NA_real_)
  }

  joint_probability <-
    contingency / n_total

  probability1 <- rowSums(
    joint_probability
  )

  probability2 <- colSums(
    joint_probability
  )

  nonzero_position <- which(
    joint_probability > 0,
    arr.ind = TRUE
  )

  mutual_information <- sum(
    joint_probability[
      nonzero_position
    ] *
      log(
        joint_probability[
          nonzero_position
        ] /
          (
            probability1[
              nonzero_position[, 1]
            ] *
              probability2[
                nonzero_position[, 2]
              ]
          )
      )
  )

  entropy1 <- -sum(
    probability1[
      probability1 > 0
    ] *
      log(
        probability1[
          probability1 > 0
        ]
      )
  )

  entropy2 <- -sum(
    probability2[
      probability2 > 0
    ] *
      log(
        probability2[
          probability2 > 0
        ]
      )
  )

  if (
    entropy1 <= 0 ||
    entropy2 <= 0
  ) {
    return(NA_real_)
  }

  mutual_information /
    sqrt(
      entropy1 * entropy2
    )
}

# ============================================================
# 5. Align bootstrap labels with the reference model
# ============================================================
align_labels_to_reference <- function(
    bootstrap_assignment,
    reference_assignment,
    K = 5L
) {

  common_assignment <-
    bootstrap_assignment %>%
    inner_join(
      reference_assignment,
      by = "source_row_id"
    )

  if (nrow(common_assignment) == 0) {
    stop(
      "Bootstrap and reference data have no observations in common."
    )
  }

  contingency <- table(
    factor(
      common_assignment$raw_module,
      levels = seq_len(K)
    ),
    factor(
      common_assignment$reference_module,
      levels = seq_len(K)
    )
  )

  assignment <- clue::solve_LSAP(
    contingency,
    maximum = TRUE
  )

  # Map each original bootstrap regime i to a reference regime.
  label_mapping <- tibble(
    raw_module = seq_len(K),
    aligned_module =
      as.integer(assignment)
  )

  common_assignment <-
    common_assignment %>%
    left_join(
      label_mapping,
      by = "raw_module"
    )

  list(
    common_assignment =
      common_assignment,

    label_mapping =
      label_mapping,

    contingency =
      contingency
  )
}

# ============================================================
# 6. Calculate Jaccard similarity for each regime
# ============================================================
calculate_module_jaccard <- function(
    data,
    module_id
) {

  bootstrap_set <- data %>%
    filter(
      aligned_module == module_id
    ) %>%
    pull(
      source_row_id
    ) %>%
    unique()

  reference_set <- data %>%
    filter(
      reference_module == module_id
    ) %>%
    pull(
      source_row_id
    ) %>%
    unique()

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

# ============================================================
# 7. Full-data reference clustering
# ============================================================
reference_model <- read_cluster_model(
  REFERENCE_JSON,
  K = K
)

reference_membership <- read_csv(
  REFERENCE_MEMBERSHIP,
  show_col_types = FALSE
)

if (
  nrow(reference_membership) !=
  length(reference_model$labels)
) {
  stop(
    "Full-data membership row count differs from the reference label count."
  )
}

reference_assignment <- tibble(
  source_row_id = as.character(
    reference_membership$source_row_id
  ),

  reference_module =
    reference_model$labels
)

# ============================================================
# 8. Find the best bootstrap model
# ============================================================
find_bootstrap_json <- function(
    bootstrap_id
) {

  candidate_paths <- c(
    file.path(
      BOOTSTRAP_RESULT_DIR,
      sprintf(
        "bootstrap_%03d",
        bootstrap_id
      ),
      "best_k5.json"
    ),

    file.path(
      BOOTSTRAP_RESULT_DIR,
      paste0(
        "bootstrap",
        bootstrap_id
      ),
      "best_k5.json"
    ),

    file.path(
      BOOTSTRAP_RESULT_DIR,
      paste0(
        "bootstrap",
        bootstrap_id,
        "_best_k5.json"
      )
    )
  )

  existing_path <- candidate_paths[
    file.exists(candidate_paths)
  ]

  if (length(existing_path) == 0) {
    return(NA_character_)
  }

  existing_path[1]
}

# ============================================================
# 9. Compare bootstrap fits with the reference model
# ============================================================
bootstrap_metric_list <- list()
bootstrap_jaccard_list <- list()
bootstrap_mapping_list <- list()
bootstrap_parameter_list <- list()
bootstrap_shared_parameter_list <- list()
aligned_assignment_list <- vector(
  mode = "list",
  length = B
)

for (b in seq_len(B)) {

  cat(
    "Analyzing bootstrap",
    b,
    "of",
    B,
    "\n"
  )

  model_path <- find_bootstrap_json(
    b
  )

  if (is.na(model_path)) {
    warning(
      "Missing bootstrap ",
      b,
      " best_k5.json"
    )
    next
  }

  membership_path <- file.path(
    BOOTSTRAP_INPUT_DIR,
    paste0(
      "bootstrap",
      b,
      "_membership.csv"
    )
  )

  if (!file.exists(membership_path)) {
    warning(
      "Missing membership file: ",
      membership_path
    )
    next
  }

  current_model <- read_cluster_model(
    model_path,
    K = K
  )

  current_membership <- read_csv(
    membership_path,
    show_col_types = FALSE
  )

  collapsed_assignment <-
    collapse_bootstrap_labels(
      membership_df = current_membership,
      labels = current_model$labels
    )

  alignment_result <-
    align_labels_to_reference(
      bootstrap_assignment =
        collapsed_assignment,

      reference_assignment =
        reference_assignment,

      K = K
    )

  common_assignment <-
    alignment_result$common_assignment

  mapping <- alignment_result$label_mapping

  # ----------------------------------------------------------
  # ARI and NMI
  # ----------------------------------------------------------
  ari_value <- mclust::adjustedRandIndex(
    common_assignment$aligned_module,
    common_assignment$reference_module
  )

  nmi_value <- calculate_nmi(
    common_assignment$aligned_module,
    common_assignment$reference_module
  )

  alignment_accuracy <- mean(
    common_assignment$aligned_module ==
      common_assignment$reference_module
  )

  bootstrap_metric_list[[b]] <- tibble(
    bootstrap_id = b,
    n_bootstrap_rows =
      nrow(current_membership),
    n_unique_bootstrap_samples =
      nrow(collapsed_assignment),
    n_common_samples =
      nrow(common_assignment),
    ARI = ari_value,
    NMI = nmi_value,
    alignment_accuracy =
      alignment_accuracy,
    inconsistent_duplicate_samples =
      sum(
        collapsed_assignment$
          number_of_assigned_modules > 1
      )
  )

  # ----------------------------------------------------------
  # Jaccard similarity for each regime
  # ----------------------------------------------------------
  bootstrap_jaccard_list[[b]] <-
    map_dfr(
      seq_len(K),
      function(module_id) {

        tibble(
          bootstrap_id = b,
          module = module_id,
          Jaccard =
            calculate_module_jaccard(
              common_assignment,
              module_id
            )
        )
      }
    )

  bootstrap_mapping_list[[b]] <-
    mapping %>%
    mutate(
      bootstrap_id = b
    )

  # Save membership labels after global alignment.
  aligned_assignment_list[[b]] <-
    common_assignment %>%
    select(
      source_row_id,
      aligned_module
    ) %>%
    distinct() %>%
    mutate(
      bootstrap_id = b
    )

  # ----------------------------------------------------------
  # Extract and align regime parameters.
  # ----------------------------------------------------------
  par <- current_model$par

  module_parameter_raw <- tibble(
    raw_module = seq_len(K),

    temperature_intercept =
      par[
        5:(4 + K)
      ],

    precipitation_intercept =
      par[
        (5 + K):(4 + 2 * K)
      ],

    temperature_slope =
      par[
        (5 + 2 * K):(4 + 3 * K)
      ],

    precipitation_slope =
      par[
        (5 + 3 * K):(4 + 4 * K)
      ],

    module_prior =
      current_model$prior_probability
  )

  bootstrap_parameter_list[[b]] <-
    module_parameter_raw %>%
    left_join(
      mapping,
      by = "raw_module"
    ) %>%
    mutate(
      bootstrap_id = b
    ) %>%
    arrange(
      aligned_module
    )

  # Four shared covariance parameters
  bootstrap_shared_parameter_list[[b]] <-
    tibble(
      bootstrap_id = b,
      temperature_phi = par[1],
      temperature_gamma = par[2],
      precipitation_phi = par[3],
      precipitation_gamma = par[4]
    )
}

bootstrap_metrics <- bind_rows(
  bootstrap_metric_list
)

bootstrap_module_jaccard <- bind_rows(
  bootstrap_jaccard_list
)

bootstrap_label_mapping <- bind_rows(
  bootstrap_mapping_list
)

bootstrap_module_parameters <- bind_rows(
  bootstrap_parameter_list
)

bootstrap_shared_parameters <- bind_rows(
  bootstrap_shared_parameter_list
)

write_csv(
  bootstrap_metrics,
  file.path(
    STABILITY_OUTPUT_DIR,
    "bootstrap_vs_reference_metrics.csv"
  )
)

write_csv(
  bootstrap_module_jaccard,
  file.path(
    STABILITY_OUTPUT_DIR,
    "bootstrap_vs_reference_module_jaccard.csv"
  )
)

write_csv(
  bootstrap_label_mapping,
  file.path(
    STABILITY_OUTPUT_DIR,
    "bootstrap_module_label_mapping.csv"
  )
)

write_csv(
  bootstrap_module_parameters,
  file.path(
    STABILITY_OUTPUT_DIR,
    "aligned_bootstrap_module_parameters.csv"
  )
)

write_csv(
  bootstrap_shared_parameters,
  file.path(
    STABILITY_OUTPUT_DIR,
    "bootstrap_shared_parameters.csv"
  )
)

# ============================================================
# 10. Percentile bootstrap summary function
# ============================================================
summarise_bootstrap_values <- function(
    values
) {

  values <- values[
    is.finite(values)
  ]

  tibble(
    n = length(values),
    mean = mean(values),
    sd = sd(values),
    median = median(values),
    CI95_lower = unname(
      quantile(
        values,
        0.025,
        na.rm = TRUE
      )
    ),
    CI95_upper = unname(
      quantile(
        values,
        0.975,
        na.rm = TRUE
      )
    )
  )
}

metric_CI <- bootstrap_metrics %>%
  select(
    ARI,
    NMI,
    alignment_accuracy
  ) %>%
  pivot_longer(
    cols = everything(),
    names_to = "metric",
    values_to = "value"
  ) %>%
  group_by(
    metric
  ) %>%
  group_modify(
    ~ summarise_bootstrap_values(
      .x$value
    )
  ) %>%
  ungroup()

print(
  metric_CI
)

write_csv(
  metric_CI,
  file.path(
    STABILITY_OUTPUT_DIR,
    "ARI_NMI_CI95.csv"
  )
)

module_jaccard_CI <-
  bootstrap_module_jaccard %>%
  group_by(
    module
  ) %>%
  group_modify(
    ~ summarise_bootstrap_values(
      .x$Jaccard
    )
  ) %>%
  ungroup()

print(
  module_jaccard_CI
)

write_csv(
  module_jaccard_CI,
  file.path(
    STABILITY_OUTPUT_DIR,
    "module_Jaccard_CI95.csv"
  )
)
