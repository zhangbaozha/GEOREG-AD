"""Real-data parity checks with frozen geometry; no full benchmark is launched."""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from georeg3dad.runtime import configure_threads, read_json, write_json
configure_threads(2)

import numpy as np
from georeg3dad.config import load_config, Interpolation, Matching
from georeg3dad.datasets import align_labels, inspect_dataset
from georeg3dad.geometry import GeoReg3DAD, TemplateLibrary, make_features, transform_xyz
from georeg3dad.scoring import anchor_scores, interpolate, residuals


def legacy(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--real-data',type=Path,required=True)
    parser.add_argument('--shape-data',type=Path,required=True)
    parser.add_argument('--runs-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    records=[]
    _,cfg=load_config(CODE/'configs/config_real3dad.json')
    raw=read_json(args.runs_root/'real3dad_local_20260922/raw_full_cases.json')
    real=inspect_dataset('real3dad',args.real_data,['candybar','gemstone'],smoke=True)
    for row in real['categories']:
        with np.load(args.runs_root/'real3dad_tuning_20260922/cache'/row['category']/'templates.npz') as lib:
            library=TemplateLibrary(lib['xyz'].copy(),lib['normals'].copy(),float(lib['h']),None,None,[])
        model=GeoReg3DAD(cfg,2);model.library=library
        for case in row['cases']:
            archived=next(c for c in raw if c['category']==row['category'] and c['sample']==case['sample'])
            scores,xyz,_=model.predict(case['test'],case['sample']+'.pcd',transform=archived['transformation'])
            with np.load(args.runs_root/'real3dad_plane_weight_20260923/predictions'/row['category']/(case['sample']+'.npz')) as expected:
                error=float(np.max(np.abs(scores-expected['u01_r06_c08_wp1'])))
                labels=align_labels(xyz,case['gt'],'real3dad',2) if case['gt'] else np.zeros(len(xyz),np.int8)
                if not np.array_equal(labels,expected['labels']) or error>1e-10:
                    raise AssertionError(('real_archive',case['sample'],error))
            records.append({'dataset':'real3dad','category':row['category'],'sample':case['sample'],
                            'comparison':'archived plane=1 scores with archived library/transform','max_error':error})
    historical=legacy('parity_legacy_pilot',CODE/'original/pilot.py')
    old_gt=legacy('parity_shape_pilot',CODE/'recovered_shapenet/pilot.py')
    _,cfg=load_config(CODE/'configs/config_shapenet.json')
    shape=inspect_dataset('shapenet',args.shape_data,['pcd__bag0','pcd__bowl0'],scope='pcd',smoke=True)
    for row in shape['categories']:
        model=GeoReg3DAD(cfg,2);lib=model.prepare(row['templates'])
        for case in row['cases']:
            scores,xyz,info=model.predict(case['test'],case['sample']+'.pcd')
            full,down,feature,_=historical.make_features(Path(case['test']),cfg.features.voxel)
            matrix=np.asarray(info['transformation'])
            query=transform_xyz(np.asarray(down.points),matrix)
            normals=np.asarray(down.normals)@matrix[:3,:3].T
            matches=historical.retrieve(query,normals,lib.xyz,lib.normals,lib.h)
            old_scores,_=historical.interpolate_to_full(transform_xyz(xyz,matrix),query,matches['raw'])
            error=float(np.max(np.abs(scores-old_scores)))
            if error>1e-10:raise AssertionError(('shape_legacy',case['sample'],error))
            if case['gt']:
                expected,_=old_gt.align_gt_by_coordinates(xyz,Path(case['gt']))
                np.testing.assert_array_equal(align_labels(xyz,case['gt'],'shapenet',2),expected)
            records.append({'dataset':'shapenet','category':row['category'],'sample':case['sample'],
                            'comparison':'historical scoring/GT with shared fresh geometry','max_error':error})
    write_json(args.output,{'status':'PASS','cases':len(records),'records':records,
        'limit':'Real3D replays archived geometry; ShapeNet compares old/new scoring on identical fresh geometry. Not a new full benchmark or Linux runtime check.'})
    print('REFACTOR_PARITY_PASS',len(records),'cases', 'max_error',max(r['max_error'] for r in records))


if __name__=='__main__':main()
