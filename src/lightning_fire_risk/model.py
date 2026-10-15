"""EM/Nelder–Mead functional mixture with two independent SAD(1) blocks.

SAD(1) convention: e_j = phi*e_(j-1) + eta_j, eta_j ~ N(0,gamma²), e_0=0.
The initial boundary and constant coefficients are explicit implementation choices.
"""
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.cluster import KMeans


def sad1_covariance(parameters, dimension=5, jitter=1e-8):
    phi, gamma = np.asarray(parameters, dtype=float)
    if not np.isfinite(phi) or not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("SAD(1) requires finite phi and positive innovation SD")
    index = np.arange(dimension)
    lag = index[:, None] - index[None, :]
    lower = np.where(lag >= 0, phi ** np.maximum(lag, 0), 0)
    return gamma**2 * (lower @ lower.T) + jitter * np.eye(dimension)


def block_means(par, k, time):
    coefficients = np.asarray(par)[4:]
    return (coefficients[:k, None] + coefficients[2*k:3*k, None] * time[None, :5],
            coefficients[k:2*k, None] + coefficients[3*k:4*k, None] * time[None, 5:])


def log_joint(values, time, par, log_weights, jitter=1e-8):
    values, time = np.asarray(values, dtype=float), np.asarray(time).reshape(-1)
    k = len(log_weights)
    means = block_means(par, k, time)
    joint = np.broadcast_to(log_weights, (len(values), k)).copy()
    for block in range(2):
        covariance = sad1_covariance(par[2*block:2*block+2], 5, jitter)
        chol = np.linalg.cholesky(covariance)
        differences = values[:, None, 5*block:5*block+5] - means[block][None, :, :]
        whitened = solve_triangular(chol, differences.reshape(-1, 5).T, lower=True)
        squared = np.sum(whitened**2, axis=0).reshape(len(values), k)
        joint -= .5 * (5*np.log(2*np.pi) + 2*np.log(np.diag(chol)).sum() + squared)
    return joint


def _profile_block(cov_parameters, design, cluster_mean, scatter, counts, jitter):
    """Exact GLS means conditional on covariance, profiling the EM objective."""
    phi, log_gamma = cov_parameters
    if not np.isfinite(cov_parameters).all() or abs(phi) > 10 or abs(log_gamma) > 20:
        return np.inf, None
    covariance = sad1_covariance([phi, np.exp(log_gamma)], 5, jitter)
    try:
        inverse = np.linalg.solve(covariance, np.eye(5))
        beta = np.linalg.solve(design.T @ inverse @ design,
                               design.T @ inverse @ cluster_mean.T).T
        residual = cluster_mean - beta @ design.T
        pooled = scatter + np.einsum("k,ki,kj->ij", counts, residual, residual)
        objective = .5 * (counts.sum()*np.linalg.slogdet(covariance)[1] + np.sum(inverse*pooled.T))
        return float(objective), beta
    except np.linalg.LinAlgError:
        return np.inf, None


def fit_mixture(values, time, k, seed, numerical):
    values, time = np.asarray(values, dtype=float), np.asarray(time).reshape(-1)
    if values.ndim != 2 or values.shape[1] != 10 or time.shape != (10,):
        raise ValueError("Expected N x 10 features and ten training indices")
    if not np.isfinite(values).all() or not np.isfinite(time).all() or not 2 <= k <= len(values):
        raise ValueError("Invalid input or regime count")
    jitter = numerical["covariance_jitter"]
    designs = [np.column_stack([np.ones(5), time[5*b:5*b+5]]) for b in range(2)]
    if any(np.linalg.matrix_rank(d) < 2 for d in designs):
        raise ValueError("Constant environmental index: slopes are not identifiable")
    labels = KMeans(n_clusters=k, n_init=numerical["kmeans_initializations"], random_state=seed).fit_predict(values)
    responsibilities = np.eye(k)[labels]
    covariance_parameters = [np.array([numerical["initial_phi"], np.log(numerical["initial_innovation_sd"])]) for _ in range(2)]
    history = []
    optimizer_success, converged = True, False
    for iteration in range(1, numerical["em_max_iterations"]+1):
        counts = responsibilities.sum(axis=0).clip(1e-12)
        log_weights = np.log(counts/counts.sum())
        coefficients, physical_covariance = [], []
        for block in range(2):
            observations = values[:, 5*block:5*block+5]
            cluster_mean = (responsibilities.T @ observations)/counts[:, None]
            scatter = observations.T @ observations - (cluster_mean*counts[:, None]).T @ cluster_mean
            scatter = (scatter+scatter.T)/2
            def objective(parameters):
                return _profile_block(parameters, designs[block], cluster_mean, scatter, counts, jitter)[0]
            initial = covariance_parameters[block]
            optimum = minimize(objective, initial, method="Nelder-Mead", options={
                "maxiter": numerical["nelder_mead_max_iterations"],
                "xatol": numerical["nelder_mead_tolerance"],
                "fatol": numerical["nelder_mead_tolerance"]})
            optimizer_success &= bool(optimum.success)
            chosen = optimum.x if np.isfinite(optimum.fun) and optimum.fun <= objective(initial) else initial
            covariance_parameters[block] = chosen
            _, beta = _profile_block(chosen, designs[block], cluster_mean, scatter, counts, jitter)
            if beta is None:
                raise RuntimeError("Non-finite Nelder–Mead M-step")
            coefficients.append(beta)
            physical_covariance.extend([float(chosen[0]), float(np.exp(chosen[1]))])
        par = np.r_[physical_covariance, coefficients[0][:, 0], coefficients[1][:, 0],
                    coefficients[0][:, 1], coefficients[1][:, 1]]
        joint = log_joint(values, time, par, log_weights, jitter)
        log_density = logsumexp(joint, axis=1)
        log_likelihood = float(log_density.sum())
        if not np.isfinite(log_likelihood):
            raise RuntimeError("Non-finite mixture likelihood")
        if history and log_likelihood < history[-1] - 1e-6*max(1, abs(history[-1])):
            raise RuntimeError("Observed likelihood decreased during EM")
        history.append(log_likelihood)
        responsibilities = np.exp(joint-log_density[:, None])
        print(f"EM {iteration}: log-likelihood={log_likelihood:.9f}")
        if len(history) > 1 and abs(history[-1]-history[-2]) <= numerical["em_tolerance"]*max(1, abs(history[-2])):
            converged = True
            break
    count_parameters = 4+4*k+k-1
    return {"cluster_number": k, "par": par.tolist(), "prob_log": log_weights.tolist(),
            "max_omega_logi": responsibilities.argmax(axis=1).tolist(),
            "log-likelihood": log_likelihood, "BIC": -2*log_likelihood+count_parameters*np.log(len(values)),
            "parameter_count": count_parameters, "iterations": iteration,
            "converged": converged, "m_step_converged": optimizer_success,
            "likelihood_history": history, "optimizer": "EM/Nelder-Mead", "covariance": "SAD1-finite-start",
            "covariance_jitter": jitter}
