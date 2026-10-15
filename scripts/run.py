"""Run the manuscript protocol configured in config/paper.json."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from threadpoolctl import threadpool_limits
from lightning_fire_risk import model, evaluation

PROTOCOL = json.loads((ROOT/"config/paper.json").read_text(encoding="utf-8"))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    threadpool_limits(limits=1)


def environment():
    return {"python": sys.version, "platform": sys.platform,
            "packages": {p: importlib.metadata.version(p) for p in
                         ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib")}}


def dump(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load_inputs(directory):
    frame = pd.read_csv(directory/"data_all.csv", index_col=0)
    time = pd.read_csv(directory/"time_all.csv", index_col=0).to_numpy().reshape(-1)
    values = frame.to_numpy(dtype=float).T
    if values.shape[1] != 10 or time.shape != (10,) or not np.isfinite(values).all() or not np.isfinite(time).all():
        raise ValueError(f"Expected finite N x 10 input and ten indices: {directory}")
    return frame, values, time


def train(directory, output, ks, dataset_id, protocol=PROTOCOL):
    frame, values, time = load_inputs(directory)
    input_hash = hashlib.sha256((directory/"data_all.csv").read_bytes()+(directory/"time_all.csv").read_bytes()).hexdigest()
    rows = []
    for k in ks:
        candidates = []
        for repeat in range(1, protocol["initializations"]+1):
            seed = protocol["seed"]+dataset_id*10000+(0 if output.parent.name=="bootstrap_models" else k*100)+repeat
            seed_all(seed)
            output.mkdir(parents=True, exist_ok=True)
            with (output/f"k{k}-{repeat}.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
                result = model.fit_mixture(values, time, k, seed, protocol["numerical"])
            result.update(seed=seed, dataset_id=dataset_id, run_id=repeat,
                feature_names=frame.index.tolist(), input_sha256=input_hash, protocol=protocol, environment=environment())
            dump(result, output/f"k{k}-{repeat}.json")
            candidates.append(result)
            rows.append({"K": k, "run": repeat, "seed": seed, "BIC": result["BIC"],
                         "NLL": -result["log-likelihood"], "converged": result["converged"]})
            print(f"{directory.name}: K={k}, run={repeat}, BIC={result['BIC']:.3f}", flush=True)
        dump(min(candidates, key=lambda x: x["BIC"]), output/f"best_k{k}.json")
    pd.DataFrame(rows).to_csv(output/"model_selection.csv", index=False)


def perturb(k, protocol=PROTOCOL):
    directory = ROOT/"results/full"
    _, values, time = load_inputs(directory)
    fitted = json.loads((directory/"json"/f"best_k{k}.json").read_text(encoding="utf-8"))
    original = evaluation.evaluate_dataset(values, time, fitted)
    labels = original.dominant_module.to_numpy()
    covariances = [model.sad1_covariance(fitted["par"][2*b:2*b+2], 5, fitted["covariance_jitter"]) for b in range(2)]
    rng = np.random.RandomState(protocol["seed"])
    output = ROOT/"results/perturbation"
    output.mkdir(parents=True, exist_ok=True)
    rows, module_rows = [], []
    for lam in protocol["perturbation_levels"]:
        records = []
        for repeat in range(1, protocol["perturbation_repeats"]+1):
            noise = np.hstack([rng.multivariate_normal(np.zeros(5), lam*cov, size=len(values)) for cov in covariances])
            result = evaluation.evaluate_dataset(values+noise, time, fitted)
            result["typicality"] = evaluation.calculate_typicality(result.NLL, original.NLL)
            result["risk"] = result.dominant_probability*result.typicality
            result["repeat"] = repeat
            result["original_module"] = labels
            pred = result.dominant_module.to_numpy()
            rows.append({"lambda": lam, "repeat": repeat, "accuracy": float(np.mean(labels==pred)),
                "ARI": adjusted_rand_score(labels, pred), "NMI": normalized_mutual_info_score(labels, pred),
                "typicality_mean": result.typicality.mean(), "risk_mean": result.risk.mean()})
            for module_id in range(1, k+1):
                selected = labels==module_id
                module_rows.append({"lambda": lam, "repeat": repeat, "module": module_id,
                    "n_original": int(selected.sum()),
                    "retention": float(np.mean(pred[selected]==module_id)) if selected.any() else np.nan})
            records.append(result)
        pd.concat(records).to_csv(output/f"risk_lambda_{lam}.csv", index=False)
    pd.DataFrame(rows).to_csv(output/"repeat_metrics.csv", index=False)
    pd.DataFrame(module_rows).to_csv(output/"module_retention.csv", index=False)
    pd.DataFrame(rows).groupby("lambda").agg(["mean", "std"]).to_csv(output/"summary.csv")
    dump({"protocol": protocol, "environment": environment()}, output/"run_metadata.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "evaluate", "perturb"])
    parser.add_argument("--group", choices=["full", "cv", "temporal", "spatial", "bootstrap"], default="full")
    args = parser.parse_args()
    seed_all(PROTOCOL["seed"])
    if args.command=="perturb":
        perturb(PROTOCOL["selected_regimes"])
        return
    if args.command=="evaluate" and args.group in ("full", "bootstrap"):
        parser.error("evaluate supports cv, temporal and spatial")
    base = ROOT/"results"/args.group
    if not base.exists():
        parser.error("Run scripts/prepare.R first")
    datasets = [base] if args.group=="full" else sorted(p for p in base.iterdir() if p.is_dir() and
        p.name.startswith(("fold_", "split_", "bootstrap_")))
    expected = {"full": 1, "cv": PROTOCOL["cv_folds"], "temporal": len(PROTOCOL["temporal_splits"]),
                "spatial": 2, "bootstrap": PROTOCOL["bootstrap_replicates"]}[args.group]
    if len(datasets)!=expected:
        parser.error(f"Expected {expected} prepared datasets, found {len(datasets)}")
    ks = PROTOCOL["candidate_regimes"] if args.group in ("full", "cv") else [PROTOCOL["selected_regimes"]]
    offset = {"full": 0, "cv": 100, "temporal": 200, "spatial": 300, "bootstrap": 0}[args.group]
    summaries = []
    for number, directory in enumerate(datasets, 1):
        output = directory/"json" if args.group!="bootstrap" else ROOT/"results/bootstrap_models"/directory.name
        if args.command=="train":
            train(directory, output, ks, 0 if args.group=="full" else offset+number)
        else:
            summaries.extend(evaluation.evaluate_split(directory, k, PROTOCOL) for k in ks)
    if summaries:
        pd.DataFrame(summaries).to_csv(base/"validation_summary.csv", index=False)


if __name__=="__main__":
    main()
