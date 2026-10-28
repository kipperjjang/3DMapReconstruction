"""Exercise packed shards through the actual training dataset and bounded cache."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from neural_scene.experiment import SequenceDataset
from neural_scene.packed import encode


@unittest.skipUnless(importlib.util.find_spec('h5py'), 'Packed storage requires h5py')
class PackedDatasetTests(unittest.TestCase):
    def test_train_reader_cache_eviction_and_path_guard(self):
        import h5py
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            sequences=[]
            points=np.array([[index*.1,0,0] for index in range(12)],dtype=np.float32)
            for split in ('train','validation','test'):
                path=directory/(split+'.h5')
                with h5py.File(str(path),'w') as file:
                    for kind in ('current','target'):
                        file[kind]=encode(points).T
                        file[kind+'_offsets']=np.arange(13)
                    for field in ('robot_pose','sensor_pose','transformation'):
                        file[field]=np.repeat(np.eye(4)[None],12,axis=0)
                    file['timestamp_ns']=np.arange(12,dtype=np.int64)*100000000
                sequences.append({'scene':split,'split':split,'shard':path.name,
                    'frames':[{'index':index,'current_points':1,'target_voxels':1} for index in range(12)]})
            manifest={'format':'neural-scene-packed-v1','input_policy':'aligned_merge_t0',
                'grid':{'voxel_size':.05,'shape':[64,64,64],'origin':[-1.6]*3},'sequences':sequences}
            (directory/'dataset.json').write_text(json.dumps(manifest))
            dataset=SequenceDataset(directory)
            dataset.cache_limit=2
            sequence=dataset.splits['train'][0]
            first=dataset.frame(sequence,0)['current_points'].copy()
            dataset.frame(sequence,1)
            dataset.frame(sequence,2)
            self.assertEqual(len(dataset.cache),2)
            self.assertNotIn(('train',0),dataset.cache)
            np.testing.assert_array_equal(dataset.frame(sequence,0)['current_points'],first)
            np.testing.assert_allclose(dataset.frame(sequence,2)['target_points'],[[.2,0,0]],atol=.0001)
            dataset.cache.clear()
            sequence['shard']='../outside.h5'
            with self.assertRaisesRegex(ValueError,'escapes'):
                dataset.frame(sequence,0)


if __name__=='__main__':
    unittest.main()
