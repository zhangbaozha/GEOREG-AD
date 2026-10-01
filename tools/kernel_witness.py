"""Small seeded real Open3D registration/scoring witness for new environments."""
from pathlib import Path
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from georeg3dad.runtime import configure_threads
configure_threads(2)
import numpy as np
import open3d as o3d
from georeg3dad.config import method_from_dict
from georeg3dad.geometry import GeoReg3DAD


def main():
    rng=np.random.default_rng(203)
    xyz=rng.normal(size=(2000,3));xyz[:,2]*=.35
    with tempfile.TemporaryDirectory(prefix='georeg-witness-') as temporary:
        path=Path(temporary)/'normal.pcd'
        assert o3d.io.write_point_cloud(str(path),o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz)))
        config=method_from_dict({'features':{'voxel':.1},'templates':{'count':1}})
        model=GeoReg3DAD(config,threads=2);library=model.prepare([path])
        scores,points,record=model.predict(path)
        assert len(scores)==2000 and np.isfinite(scores).all() and record['registration']['icp_fitness']>.99
        print(f'WITNESS_PASS anchors={len(library.xyz)} full_points={len(points)} device=CPU Open3D={o3d.__version__}')


if __name__=='__main__':main()
