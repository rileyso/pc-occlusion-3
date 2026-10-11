import unittest

import numpy as np

from dairv2x.tools.cache_sequence import select_sequence
from dairv2x.tools.coalign import result_record


class SequenceTest(unittest.TestCase):
    def test_timestamp_order_split_filter_gaps_and_result_schema(self):
        pairs = [{'vehicle_pointcloud_path': f'vehicle-side/velodyne/{i}.pcd',
                  'infrastructure_pointcloud_path': f'infrastructure-side/velodyne/{i}.pcd'}
                 for i in ('a', 'b', 'c')]
        rows = [{'pointcloud_path': f'velodyne/{i}.pcd', 'batch_id': '106',
                 'pointcloud_timestamp': str(t), 'intersection_loc': 'road'}
                for i, t in (('a', 200000), ('b', 100000), ('c', 300000))]
        frames = select_sequence(pairs, rows, rows, ['a', 'b'], '106')
        self.assertEqual([f['vehicle_frame_id'] for f in frames], ['b', 'a'])
        self.assertEqual([f['pair_index'] for f in frames], [1, 0])
        self.assertEqual(frames[0]['prediction_file'], 'b.json')
        with self.assertRaisesRegex(ValueError, 'No paired'):
            select_sequence(pairs, rows, rows, ['a'], 'other')
        with self.assertRaisesRegex(ValueError, 'Timestamp gap'):
            select_sequence(pairs, rows, rows, ['a', 'b'], '106', max_gap=0.05)
        wrong_infrastructure = [dict(r, batch_id=r['pointcloud_path']) for r in rows]
        with self.assertRaisesRegex(ValueError, 'crosses'):
            select_sequence(pairs, rows, wrong_infrastructure, ['a', 'b'], '106')
        result = {'gt_boxes': np.zeros((1, 8, 3)), 'pred_boxes': np.ones((1, 8, 3)),
                  'scores': np.array([0.9]), 'matches': [None]}
        record = result_record(1, pairs[1], 'val', 0.5, result)
        self.assertEqual(record['vehicle_frame_id'], 'b')
        self.assertEqual(record['pair_index'], 1)
        self.assertEqual(record['split'], 'val')
        self.assertIsNone(record['ground_truth'][0]['coalign'])
        self.assertEqual(record['predictions']['coalign'][0]['confidence'], 0.9)
        self.assertEqual(record['predictions']['coalign'][0]['corners_vehicle_lidar'], np.ones((8, 3)).tolist())
        single_record = result_record(1, pairs[1], 'val', 0.5, result,
                                      prediction_key='vehicle_only')
        self.assertIn('vehicle_only', single_record['predictions'])
        self.assertIsNone(single_record['ground_truth'][0]['vehicle_only'])


if __name__ == '__main__':
    unittest.main()
