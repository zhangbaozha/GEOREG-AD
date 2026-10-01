import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace

from georeg3dad.runtime import configure_threads
configure_threads(1)

import numpy as np
from scipy.spatial.distance import cdist

from georeg3dad.config import Interpolation, Matching, load_config, method_from_dict
from georeg3dad.datasets import align_labels, canonical_filename, point_count
from georeg3dad.geometry import GeoReg3DAD, registration_seed
from georeg3dad.runtime import available_memory, read_json, write_json
from georeg3dad.scoring import anchor_scores, interpolate, residuals

CODE = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def test_separate_profiles_and_override(self):
        _, real = load_config(CODE/'configs/config_real3dad.json')
        _, shape = load_config(CODE/'configs/config_shapenet.json')
        _, balanced = load_config(CODE/'configs/config_real3dad.json', 'balanced')
        _, baseline = load_config(CODE/'configs/config_real3dad.json', 'baseline')
        self.assertEqual((real.interpolation.k, real.matching.plane_weight), (128, 1))
        self.assertEqual((shape.features.voxel, shape.interpolation.k), (.05, 3))
        self.assertEqual(balanced.matching.plane_weight, .5)
        self.assertEqual((baseline.interpolation.k, baseline.matching.unmatched_penalty), (3, 16))
        _, changed = load_config(CODE/'configs/config_real3dad.json', overrides=['matching.normal_weight=0.25'])
        self.assertEqual(changed.matching.normal_weight, .25)

    def test_invalid_config_fails_early(self):
        for data in ({'bad':{}}, {'matching':{'typo':1}}, {'features':{'voxel':0}},
                     {'interpolation':{'k':2.5}}, {'matching':{'plane_weight':float('nan')}},
                     {'templates':{'reference_index':4}}, {'object_top_fraction':2},
                     {'interpolation':{'k':True}}, {'features':{'normal_radius':6}}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                method_from_dict(data)


class ScoringTests(unittest.TestCase):
    def test_against_brute_force_geometry(self):
        rng = np.random.default_rng(12)
        query = rng.normal(size=(17,3)); query[-1] = 100
        template = rng.normal(size=(23,3))
        qn = rng.normal(size=query.shape); tn = rng.normal(size=template.shape)
        qn /= np.linalg.norm(qn,axis=1)[:,None]; tn /= np.linalg.norm(tn,axis=1)[:,None]
        for k in (1,8,32):
            for weight in (0,.25,.5,1):
                cfg = Matching(candidate_k=k,radius_h=3,plane_weight=weight,unmatched_penalty=1)
                components = residuals(query,qn,template,tn,.7,cfg)
                actual, unmatched = anchor_scores(components,cfg)
                expected = []
                for point, normal in zip(query,qn):
                    ds = np.linalg.norm(template-point,axis=1)/.7
                    ids = np.argsort(ds)[:k]; ids = ids[ds[ids]<3]
                    if len(ids):
                        cost = ds[ids]+weight*np.abs(np.sum((point-template[ids])*tn[ids],axis=1))/.7+.5*(1-np.abs(tn[ids]@normal))
                        expected.append(cost.min())
                    else:
                        expected.append(1)
                np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
                self.assertTrue(unmatched[-1])

    def test_weight_reselects_candidate_and_radius_is_strict(self):
        components = {'distance':np.array([[1.,1.5],[6.,7.]]),
                      'plane':np.array([[2.,.1],[0.,0.]]), 'normal':np.zeros((2,2)),
                      'search_distance':np.array([[1.,1.5],[6.,7.]])}
        for weight, expected in ((.25,1.5),(.5,1.55),(1,1.6)):
            actual, missing = anchor_scores(components,Matching(candidate_k=2,radius_h=6,plane_weight=weight,unmatched_penalty=1))
            np.testing.assert_allclose(actual,[expected,1])
            np.testing.assert_array_equal(missing,[False,True])

    def test_interpolation_with_chunks_small_cloud_and_exact_match(self):
        rng=np.random.default_rng(4)
        anchors=rng.normal(size=(11,3)); full=np.vstack([anchors[:1],rng.normal(size=(19,3))]); values=rng.random(11)
        for k in (1,3,128):
            for power in (0,1,2):
                cfg=Interpolation(k=k,power=power,chunk_size=3)
                distances=cdist(full,anchors); ids=np.argsort(distances,axis=1)[:,:min(k,11)]
                if power==0:
                    expected=values[ids].mean(axis=1)
                else:
                    ds=np.take_along_axis(distances,ids,axis=1)
                    weights=np.maximum(ds,1e-8)**(-power); weights/=weights.sum(axis=1)[:,None]
                    expected=(values[ids]*weights).sum(axis=1)
                np.testing.assert_allclose(interpolate(full,anchors,values,cfg),expected,rtol=0,atol=1e-12)


class DatasetAndPortabilityTests(unittest.TestCase):
    def test_seed_independent_of_native_path_and_shapenet_name(self):
        shape=canonical_filename('shapenet','bag0_positive0.pcd')
        self.assertEqual(shape,'bag0_good0.pcd')
        self.assertEqual(registration_seed('C:\\data\\test\\'+shape), registration_seed('/data/test/'+shape))

    def test_gt_mapping_rounding_and_invalid_labels(self):
        rng=np.random.default_rng(8); xyz=rng.normal(size=(25,3)); y=(xyz[:,0]>0).astype(int); order=rng.permutation(25)
        with tempfile.TemporaryDirectory(prefix='georeg space ') as directory:
            path=Path(directory)/'标签.txt'
            for dataset,delimiter in (('real3dad',' '),('shapenet',',')):
                np.savetxt(path,np.column_stack([xyz[order],y[order]]),delimiter=delimiter)
                np.testing.assert_array_equal(align_labels(xyz,path,dataset),y)
            bad=np.column_stack([xyz,y.astype(float)]);bad[0,3]=.5
            np.savetxt(path,bad)
            with self.assertRaises(ValueError):align_labels(xyz,path,'real3dad')
            np.savetxt(path,np.column_stack([xyz[:5],y[:5]]))
            with self.assertRaises(ValueError):align_labels(xyz,path,'real3dad')

    def test_atomic_json_and_memory(self):
        with tempfile.TemporaryDirectory(prefix='georeg space ') as directory:
            path=Path(directory)/'参数.json'
            write_json(path,{'a':1});write_json(path,{'a':2})
            self.assertEqual(read_json(path),{'a':2})
        self.assertGreater(available_memory(),0)

    def test_small_end_to_end_point_cloud(self):
        import open3d as o3d
        rng=np.random.default_rng(9);xyz=rng.uniform(-1,1,(900,3));xyz[:,2]*=.3
        with tempfile.TemporaryDirectory(prefix='georeg space ') as directory:
            path=Path(directory)/'normal.pcd'
            o3d.io.write_point_cloud(str(path),o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz)))
            self.assertEqual(point_count(path),len(xyz))
            cfg=method_from_dict({'features':{'voxel':.1},'templates':{'count':1},'interpolation':{'k':3}})
            model=GeoReg3DAD(cfg);model.prepare([path])
            scores,read_xyz,record=model.predict(path,transform=np.eye(4))
            self.assertEqual(len(scores),900)
            self.assertLess(float(scores.max()),1e-7)
            self.assertEqual(record['registration'],{'replayed':True})


if __name__=='__main__':unittest.main()
