"""Observed mixture density, empirical typicality and bootstrap intervals."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from .model import log_joint


def evaluate_dataset(values, time, fitted, sample_ids=None):
    joint = log_joint(values, time, fitted["par"], fitted["prob_log"], fitted["covariance_jitter"])
    density = logsumexp(joint, axis=1)
    posterior = np.exp(joint-density[:, None])
    k = posterior.shape[1]
    frame = pd.DataFrame({"sample_id": np.arange(len(values)) if sample_ids is None else sample_ids,
        "log_mixture_density": density, "NLL": -density,
        "dominant_module": posterior.argmax(axis=1)+1,
        "dominant_probability": posterior.max(axis=1),
        "normalized_posterior_entropy": -(posterior*np.log(posterior.clip(1e-300))).sum(axis=1)/np.log(k)})
    for c in range(k):
        frame[f"M{c+1}_posterior"] = posterior[:, c]
    return frame


def calculate_typicality(test_nll, reference_nll):
    reference = np.sort(np.asarray(reference_nll, dtype=float))
    if not len(reference) or not np.isfinite(reference).all():
        raise ValueError("Typicality requires a nonempty finite reference distribution")
    return (len(reference)-np.searchsorted(reference, test_nll, side="left"))/len(reference)


def bootstrap_ci(values, statistic=np.mean, n_bootstrap=5000, confidence_level=.95, random_seed=2026):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Confidence intervals require finite observations")
    rng = np.random.default_rng(random_seed)
    statistics = [statistic(rng.choice(values, size=len(values), replace=True)) for _ in range(n_bootstrap)]
    alpha = (1-confidence_level)/2
    return np.quantile(statistics, [alpha, 1-alpha]).tolist()


def evaluate_split(directory, k, protocol):
    directory = Path(directory)
    fitted = json.loads((directory/"json"/f"best_k{k}.json").read_text(encoding="utf-8"))
    time = pd.read_csv(directory/"time_all.csv", index_col=0).to_numpy().reshape(-1)
    train_matrix = pd.read_csv(directory/"data_all.csv", index_col=0)
    test_matrix = pd.read_csv(directory/"test_data_all.csv", index_col=0)
    if not train_matrix.index.equals(test_matrix.index):
        raise ValueError("Held-out feature order differs from training order")
    train = evaluate_dataset(train_matrix.to_numpy().T, time, fitted, train_matrix.columns)
    test = evaluate_dataset(test_matrix.to_numpy().T, time, fitted, test_matrix.columns)
    test["typicality"] = calculate_typicality(test.NLL, train.NLL)
    test["risk"] = test.typicality*test.dominant_probability
    numerical = protocol["numerical"]
    summary = {"split": directory.name, "K": k, "n_train": len(train), "n_test": len(test),
               "mean_NLL": float(test.NLL.mean()), "median_typicality": float(test.typicality.median()),
               "mean_typicality": float(test.typicality.mean()), "reference": "training_in_sample_NLL"}
    for offset, metric in enumerate(["NLL", "typicality"]):
        low, high = bootstrap_ci(test[metric], n_bootstrap=numerical["ci_bootstrap_draws"],
            confidence_level=numerical["confidence_level"], random_seed=protocol["seed"]+offset)
        summary[f"mean_{metric}_CI95_lower"], summary[f"mean_{metric}_CI95_upper"] = low, high
    output = directory/f"evaluation_k{k}"
    output.mkdir(parents=True, exist_ok=True)
    train.to_csv(output/"train_sample_metrics.csv", index=False)
    test.to_csv(output/"test_sample_metrics.csv", index=False)
    (output/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
