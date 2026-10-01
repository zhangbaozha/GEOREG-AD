"""Small real subprocess test: outputs, resume, source inputs and corruption gates."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

from georeg3dad.runtime import configure_threads
configure_threads(1)
import numpy as np
import open3d as o3d

from georeg3dad.config import method_from_dict
from georeg3dad.runner import run, verify_category
from georeg3dad.runtime import read_json


class RunnerTests(unittest.TestCase):
    def test_subprocess_resume_and_integrity(self):
        rng=np.random.default_rng(29)
        xyz=rng.uniform(-1,1,(600,3));xyz[:,2]*=.4
        with tempfile.TemporaryDirectory(prefix='georeg runner ') as directory:
            root=Path(directory);data=root/'data';data.mkdir();output=root/'results'
            def cloud(name,points):
                path=data/name
                self.assertTrue(o3d.io.write_point_cloud(str(path),o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))))
                return str(path)
            template=cloud('template.pcd',xyz)
            normal=cloud('normal.pcd',xyz)
            changed=xyz.copy();changed[100:150,2]+=.1
            anomaly=cloud('anomaly.pcd',changed)
            gt=data/'anomaly.txt';labels=np.zeros(600);labels[100:150]=1
            np.savetxt(gt,np.column_stack([changed,labels]),fmt='%.17g')
            cases=[{'test':normal,'gt':None,'is_anomaly':False,'point_gt_valid':True,'sample':'normal','points':600},
                   {'test':anomaly,'gt':str(gt),'is_anomaly':True,'point_gt_valid':True,'sample':'anomaly','points':600}]
            row={'category':'synthetic','source_split':'real3dad','source_category':'synthetic','templates':[template],'cases':cases}
            manifest={'dataset':'real3dad','data_root':str(data),'scope':'all','smoke':True,'categories':[row],'test_scans':2,'point_valid_scans':2}
            cfg=method_from_dict({'features':{'voxel':.1},'templates':{'count':1}})
            settings={'dataset':'real3dad','method':asdict(cfg)}
            run(deepcopy(manifest),settings,output,workers=1,threads=1)
            self.assertEqual(read_json(output/'status.json')['state'],'COMPLETE')
            self.assertEqual(read_json(output/'verification.json')['status'],'PASS')
            prediction=output/'results/synthetic/scores/anomaly.npz'
            stamp=prediction.stat().st_mtime_ns
            run(deepcopy(manifest),settings,output,workers=1,threads=1,resume=True)
            self.assertEqual(prediction.stat().st_mtime_ns,stamp,'Resume should reuse verified categories')
            with self.assertRaises(ValueError):run(deepcopy(manifest),settings,output,workers=1,threads=1)
            prediction.write_bytes(b'corrupted test output')
            with self.assertRaises(ValueError):verify_category(output,row)
            altered=deepcopy(settings);altered['method']['matching']['plane_weight']=1
            previous_status=(output/'status.json').read_bytes()
            with self.assertRaises(ValueError):run(deepcopy(manifest),altered,output,workers=1,threads=1,resume=True)
            self.assertEqual((output/'status.json').read_bytes(),previous_status)


if __name__=='__main__':unittest.main()
