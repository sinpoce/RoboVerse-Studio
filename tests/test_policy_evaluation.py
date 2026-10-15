import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from stagewise_evaluate import BackflipTracker


class BackflipCriteriaTests(unittest.TestCase):
    def test_standing_and_half_turn_do_not_count_as_success(self):
        for rotation in (0, -math.pi):
            tracker = BackflipTracker(0.34)
            tracker.update(0.1, rotation/0.1, 0.6, -1, 0, False, 0, 0)
            for _ in range(10): tracker.update(0.05, 0, 0.34, 1, 4, False, 0, 0)
            self.assertFalse(tracker.result()["success"])

    def test_backward_flight_and_stable_landing_succeed(self):
        tracker = BackflipTracker(0.34)
        tracker.update(0.1, -2*math.pi/0.1, 0.6, -1, 0, False, 0, 0)
        for _ in range(6): tracker.update(0.05, 0, 0.34, 1, 4, False, 0.1, 0.2)
        self.assertTrue(tracker.result()["success"])

    def test_forward_flip_and_body_supported_landing_fail(self):
        for direction, body_contact in ((1, False), (-1, True)):
            tracker = BackflipTracker(0.34)
            tracker.update(0.1, direction*2*math.pi/0.1, 0.6, -1, 0, False, 0, 0)
            for _ in range(10): tracker.update(0.05, 0, 0.34, 1, 4, body_contact, 0, 0)
            self.assertFalse(tracker.result()["success"])
