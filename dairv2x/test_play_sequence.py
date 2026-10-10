import unittest

import numpy as np

from dairv2x.play_sequence import frame_index_at


class PlaySequenceTest(unittest.TestCase):
    def test_frame_index_at(self):
        ts = np.array([100_000, 200_000, 300_000], dtype=np.int64)
        self.assertEqual(frame_index_at(ts, 0), 0)
        self.assertEqual(frame_index_at(ts, 50_000), 0)
        self.assertEqual(frame_index_at(ts, 100_000), 1)
        self.assertEqual(frame_index_at(ts, 150_000), 1)
        self.assertEqual(frame_index_at(ts, 1_000_000), 2)


if __name__ == "__main__":
    unittest.main()
