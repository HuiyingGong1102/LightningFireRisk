"""Scientific invariants for the manuscript model; study data are not required."""
import json
from pathlib import Path
import sys
import unittest
import contextlib
import io
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from lightning_fire_risk import model, evaluation


class ModelChecks(unittest.TestCase):
    def test_sad1_recurrence(self):
        covariance = model.sad1_covariance([.5, .2], 5, jitter=0)
        expected_variance = .04*np.cumsum(.5**(2*np.arange(5)))
        np.testing.assert_allclose(np.diag(covariance), expected_variance)
        self.assertTrue((np.linalg.eigvalsh(covariance)>0).all())
        self.assertNotEqual(covariance[0, 0], covariance[-1, -1])

    def test_em_likelihood_and_reproducibility(self):
        protocol = json.loads((ROOT/"config/paper.json").read_text(encoding="utf-8"))["numerical"]
        protocol["em_max_iterations"] = 20
        rng = np.random.RandomState(42)
        time = np.tile(np.linspace(.1, .5, 5), 2)
        values = np.vstack([.3+.2*time+rng.normal(0, .02, (25, 10)),
                            .7-.1*time+rng.normal(0, .02, (25, 10))])
        fits = []
        with threadpool_limits(limits=1), contextlib.redirect_stdout(io.StringIO()):
            for _ in range(2):
                fits.append(model.fit_mixture(values, time, 2, 2026, protocol))
        np.testing.assert_array_equal(fits[0]["par"], fits[1]["par"])
        self.assertTrue(np.all(np.diff(fits[0]["likelihood_history"])>=-1e-6))
        scores = evaluation.evaluate_dataset(values, time, fits[0])
        np.testing.assert_allclose(scores[["M1_posterior", "M2_posterior"]].sum(axis=1), 1)
        self.assertAlmostEqual(fits[0]["log-likelihood"], -scores.NLL.sum())
        self.assertAlmostEqual(fits[0]["BIC"], 2*scores.NLL.sum()+13*np.log(len(values)))

    def test_typicality_and_risk(self):
        typicality = evaluation.calculate_typicality([-100, 1, 100], [0, 1, 2])
        np.testing.assert_allclose(typicality, [1, 2/3, 0])
        risk = typicality*np.array([.8, .6, .5])
        self.assertTrue(np.all((risk>=0) & (risk<=1)))


if __name__=="__main__":
    unittest.main()
