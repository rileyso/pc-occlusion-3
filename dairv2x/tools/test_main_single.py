import unittest

from dairv2x.main_single import annotation_key, pascal_difficulty


class PascalDifficultyTest(unittest.TestCase):
    def test_assigns_easiest_matching_difficulty(self):
        self.assertEqual(pascal_difficulty(self.label(40, 0, 0)), "easy")
        self.assertEqual(pascal_difficulty(self.label(39, 0, 0)), "moderate")
        self.assertEqual(pascal_difficulty(self.label(25, 1, None)), "moderate")
        self.assertEqual(pascal_difficulty(self.label(25, 2, None)), "hard")
        self.assertIsNone(pascal_difficulty(self.label(24, 0, 0)))

    def test_annotation_key_separates_camera_sources(self):
        vehicle = self.label(40, 0, 0)
        infrastructure = self.label(40, 0, 0)
        infrastructure["2d_box"]["xmin"] = 20
        self.assertNotEqual(annotation_key(vehicle), annotation_key(infrastructure))

    @staticmethod
    def label(height, occlusion, truncation):
        return {
            "2d_box": {"ymin": 10, "ymax": 10 + height},
            "occluded_state": occlusion,
            "truncated_state": truncation,
        }


if __name__ == "__main__":
    unittest.main()
