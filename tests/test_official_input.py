"""Official geometry source, full coverage and label-isolation regressions."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from georeg3dad.runtime import configure_threads
configure_threads(1)
import numpy as np
import open3d as o3d
from georeg3dad.config import method_from_dict
from georeg3dad.datasets import inspect_dataset, official_labels
from georeg3dad.geometry import read_geometry, GeoReg3DAD
from georeg3dad.scoring import interpolate


class OfficialInputTests(unittest.TestCase):
    def test_txt_rows_survive_pcd_mismatch_and_label_changes_do_not_change_scores(self):
        rng = np.random.default_rng(81)
        xyz = rng.normal(size=(700, 3)); xyz[:, 2] *= .2
        labels = np.arange(700) % 2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); folder = root / 'synthetic'
            for name in ('train', 'test', 'gt'):
                (folder / name).mkdir(parents=True)
            def pcd(path, points):
                self.assertTrue(o3d.io.write_point_cloud(str(path), o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))))
            for i in range(4):
                pcd(folder / 'train' / f'template{i}.pcd', xyz)
            pcd(folder / 'test' / '1_good.pcd', xyz)
            pcd(folder / 'test' / '2_sink.pcd', np.vstack([xyz, [[9, 9, 9]]]))
            gt = folder / 'gt' / '2_sink.txt'
            np.savetxt(gt, np.column_stack([xyz, labels]))
            protocol = dict(categories=[dict(category='synthetic', source_category='synthetic',
                source_split='real3dad', test_scans=2, normal_scans=1, point_valid_scans=1)],
                exclusions=[dict(category='synthetic', sample='2_sink', sha256='unused')])
            with patch('georeg3dad.datasets.protocol', return_value=protocol):
                manifest = inspect_dataset('real3dad', root, input_protocol='real3dad-official')
            self.assertEqual(manifest['point_valid_scans'], 2)
            case = manifest['categories'][0]['cases'][1]
            self.assertEqual(case['input'], str(gt)); self.assertEqual(case['points'], 700)
            np.testing.assert_array_equal(official_labels(case), labels)
            np.testing.assert_allclose(np.asarray(read_geometry(gt, True).points), xyz - xyz.mean(axis=0), atol=1e-14)
            cfg = method_from_dict({'features': {'voxel': .1}, 'templates': {'count': 1}, 'interpolation': {'k': 128, 'power': 0}})
            model = GeoReg3DAD(cfg, center=True); model.prepare([folder / 'train/template0.pcd'])
            first, _, _, parts = model.predict(gt, transform=np.eye(4), return_intermediates=True)
            np.testing.assert_array_equal(first, interpolate(parts['registered_xyz'], parts['registered_anchors'], parts['raw_scores'], cfg.interpolation))
            np.savetxt(gt, np.column_stack([xyz, 1 - labels]))
            second, _, _ = model.predict(gt, transform=np.eye(4))
            np.testing.assert_array_equal(first, second)
            np.testing.assert_array_equal(official_labels(case), 1 - labels)
            np.savetxt(gt, np.column_stack([xyz, np.full(700, .5)]))
            with self.assertRaises(ValueError):
                official_labels(case)


if __name__ == '__main__':
    unittest.main()
