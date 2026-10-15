# Event-level nonlinear least squares: initialize with log–log regression,
# then minimize residual sum of squares on the original response scale.
power_equation_base2 <- function(x, y, control) {
  if(any(!is.finite(c(x,y))) || any(x<=0) || any(y<=0))
    stop("Power-law fitting requires finite, strictly positive values")
  coefficients <- coef(lm(log(y)~log(x)))
  initial <- c(exp(coefficients[1]),coefficients[2])
  objective <- function(parameters) sum((y-parameters[1]*x^parameters[2])^2)
  fit <- optim(initial,objective,method="Nelder-Mead",control=control)
  fit$par
}

power_equation_fit2 <- function(x,y,X_smooth,thread=1L) {
  control <- list(maxit=protocol$numerical$nelder_mead_max_iterations,
                  reltol=protocol$numerical$nelder_mead_tolerance)
  if(thread<=1) {
    parameters <- lapply(seq_len(ncol(y)),function(i) power_equation_base2(x,y[,i],control))
  } else {
    workers <- parallel::makeCluster(thread)
    on.exit(parallel::stopCluster(workers),add=TRUE)
    parallel::clusterExport(workers,c("power_equation_base2","x","y","control"),envir=environment())
    parameters <- parallel::parLapply(workers,seq_len(ncol(y)),function(i)
      power_equation_base2(x,y[,i],control))
  }
  fitted <- vapply(parameters,function(p) p[1]*x^p[2],numeric(length(x)))
  smooth <- vapply(parameters,function(p) p[1]*X_smooth^p[2],numeric(length(X_smooth)))
  list(original_data=y,power_par=parameters,power_fit=fitted,Time=x,smooth_fit=smooth)
}
