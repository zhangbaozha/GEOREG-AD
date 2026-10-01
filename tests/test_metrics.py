import unittest
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from georeg3dad.metrics import point_metrics


class RankingMetricsTests(unittest.TestCase):
    def test_sklearn_parity_with_ties_and_chunk_boundaries(self):
        rng = np.random.default_rng(173)
        y = rng.integers(0, 2, 1003, dtype=np.int8)
        for s in (rng.normal(size=len(y)), rng.integers(0, 7, len(y)),
                  np.ones(len(y)), y.astype(float), 1-y.astype(float)):
            for chunk in (1, 7, 64, 1024):
                actual = point_metrics(y, s, chunk)
                self.assertAlmostEqual(actual['p_auroc'], roc_auc_score(y, s), places=13)
                self.assertAlmostEqual(actual['p_ap'], average_precision_score(y, s), places=13)

    def test_invalid_inputs(self):
        for y,s in (([0,0],[1,2]), ([0,2],[1,2]), ([0,1],[1,np.nan])):
            with self.assertRaises(ValueError):
                point_metrics(np.array(y), np.array(s))


if __name__ == '__main__':
    unittest.main()
