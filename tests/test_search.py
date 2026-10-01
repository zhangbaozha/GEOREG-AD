from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
spec=importlib.util.spec_from_file_location('shape_search',CODE/'tools/tune_shapenet.py')
search=importlib.util.module_from_spec(spec);spec.loader.exec_module(search)
from georeg3dad.config import load_config


class SearchTests(unittest.TestCase):
    def test_all_stage_grids_keep_controls_and_geometry(self):
        settings,_=load_config(CODE/'configs/config_shapenet.json')
        base=search.setting(settings['method'])
        for stage in search.STAGES:
            configs=search.grid(stage,base,base)
            self.assertEqual(configs[0],base)
            self.assertEqual(len(configs),len({c['id'] for c in configs}))
            for c in configs:
                for field in ('features','registration','templates'):
                    self.assertEqual(c['method'][field],base['method'][field])

    def test_selection_rejects_one_metric_regression_and_keeps_incumbent(self):
        incumbent={'id':'base'}
        rows=[{'config':'base','p_auroc':.8,'p_ap':.4},
              {'config':'bad_auc','p_auroc':.79,'p_ap':.6},
              {'config':'bad_ap','p_auroc':.9,'p_ap':.39}]
        self.assertEqual(search.choose({'metrics':rows},incumbent),'base')
        rows.append({'config':'both','p_auroc':.81,'p_ap':.42})
        self.assertEqual(search.choose({'metrics':rows},incumbent),'both')

    def test_split_determinism_and_variant_group_isolation(self):
        cases=[]
        for i in range(15):
            for variant in ('good','bulge','concavity'):
                cases.append({'sample':f'bag0_{variant}{i}','is_anomaly':variant!='good','point_gt_valid':True})
        manifest={'categories':[{'category':'pcd__bag0','cases':cases}]}
        splits=search.make_splits(manifest)
        shuffled=deepcopy(manifest);shuffled['categories'][0]['cases'].reverse()
        self.assertEqual(splits,search.make_splits(shuffled))
        for i in range(15):
            self.assertEqual(len({splits['pcd__bag0'][f'bag0_{variant}{i}'] for variant in ('good','bulge','concavity')}),1)


if __name__=='__main__':unittest.main()
