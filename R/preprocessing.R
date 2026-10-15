fit_minmax_1_2 <- function(x) {
    x <- suppressWarnings(as.numeric(as.character(x)))
    x_valid <- x[is.finite(x)]
    if (length(x_valid) == 0) {
        return(list(method = "minmax_1_2", min = NA_real_, max = NA_real_, constant = TRUE))
    }
    x_min <- min(x_valid)
    x_max <- max(x_valid)
    list(method = "minmax_1_2", min = x_min, max = x_max, constant = isTRUE(all.equal(x_min, x_max)))
}

fit_log_minmax_reverse_1_2 <- function(x, lower_q = 0.01, upper_q = 0.99) {
    x <- suppressWarnings(as.numeric(as.character(x)))
    if (any(x < 0, na.rm = TRUE)) {
        stop("There are negative values in the precipitation variables, and log1p cannot be directly applied.")
    }
    x_log <- log1p(x)
    x_valid <- x_log[is.finite(x_log)]
    if (length(x_valid) == 0) {
        return(list(method = "log_minmax_reverse_1_2", lower_q = lower_q, upper_q = upper_q, q_low = NA_real_,
            q_high = NA_real_, constant = TRUE))
    }
    q_low <- as.numeric(quantile(x_valid, probs = lower_q, na.rm = TRUE, names = FALSE))
    q_high <- as.numeric(quantile(x_valid, probs = upper_q, na.rm = TRUE, names = FALSE))
    list(method = "log_minmax_reverse_1_2", lower_q = lower_q, upper_q = upper_q, q_low = q_low, q_high = q_high,
        constant = isTRUE(all.equal(q_low, q_high)))
}

transform_minmax_1_2 <- function(x, param, clip = FALSE) {
    x <- suppressWarnings(as.numeric(as.character(x)))
    if (isTRUE(param$constant) || !is.finite(param$min) || !is.finite(param$max)) {
        result <- rep(1, length(x))
        result[is.na(x)] <- NA_real_
        return(result)
    }
    if (clip) {
        x <- pmin(pmax(x, param$min), param$max)
    }
    x_scaled <- (x - param$min)/(param$max - param$min)
    1 + x_scaled
}

transform_log_minmax_reverse_1_2 <- function(x, param) {
    x <- suppressWarnings(as.numeric(as.character(x)))
    if (any(x < 0, na.rm = TRUE)) {
        stop("There are negative values in the precipitation variables, and log1p cannot be directly applied.")
    }
    if (isTRUE(param$constant) || !is.finite(param$q_low) || !is.finite(param$q_high)) {
        result <- rep(1, length(x))
        result[is.na(x)] <- NA_real_
        return(result)
    }
    x_log <- log1p(x)
    x_clip <- pmin(pmax(x_log, param$q_low), param$q_high)
    x_scaled <- (x_clip - param$q_low)/(param$q_high - param$q_low)
    1 + (1 - x_scaled)
}

fit_fire_scaler <- function(train_data, temp_features, rain_features, lower_q = 0.01, upper_q = 0.99) {
    all_features <- c(temp_features, rain_features)
    missing_features <- setdiff(all_features, names(train_data))
    if (length(missing_features) > 0) {
        stop("The training data is lacking variables:", paste(missing_features,
            collapse = "、"))
    }
    params <- list(temp_features = temp_features, rain_features = rain_features, lower_q = lower_q, upper_q = upper_q,
        variables = list())
    for (feature in temp_features) {
        params$variables[[feature]] <- fit_minmax_1_2(train_data[[feature]])
    }
    for (feature in rain_features) {
        params$variables[[feature]] <- fit_log_minmax_reverse_1_2(train_data[[feature]], lower_q = lower_q,
            upper_q = upper_q)
    }
    return(params)
}

transform_fire_data <- function(data, scaler, clip_temperature = FALSE, suffix = "_index") {
    result <- data
    for (feature in scaler$temp_features) {
        if (!feature %in% names(result)) {
            stop("The training data is lacking variables:", feature)
        }
        result[[paste0(feature, suffix)]] <- transform_minmax_1_2(x = result[[feature]], param = scaler$variables[[feature]],
            clip = clip_temperature)
    }
    for (feature in scaler$rain_features) {
        if (!feature %in% names(result)) {
            stop("The training data is lacking variables:", feature)
        }
        result[[paste0(feature, suffix)]] <- transform_log_minmax_reverse_1_2(x = result[[feature]],
            param = scaler$variables[[feature]])
    }
    return(result)
}
