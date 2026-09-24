"""Guard the heap evidence against omitted tail sites and bad totals."""
import json
from pathlib import Path
import tempfile
import unittest

from profile_model_heap_followup import compare_peak, summarize_profile


class HeapEvidenceTest(unittest.TestCase):
    def test_includes_thirteenth_site_and_checks_end_heap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.dhat.json"
            log = path.with_suffix(".log")
            points = [{"tb": 1000, "tbk": 1, "gb": 1000, "gbk": 1,
                       "eb": 0, "ebk": 0, "fs": [0]} for _ in range(12)]
            points.append({"tb": 49, "tbk": 3, "gb": 49, "gbk": 3,
                           "eb": 1, "ebk": 1, "fs": [1]})
            path.write_text(json.dumps({"pps": points, "ftbl": [
                "0x1234: retained_history (history.rs:10)",
                "0x5678: tail_allocation (small.rs:20)"]}))
            log.write_text("Total: 12,049 bytes in 15 blocks\n"
                           "At t-gmax: 12,049 bytes in 15 blocks\n"
                           "At t-end: 1 bytes in 1 blocks\n")
            frames = {}
            result = summarize_profile(path, log, frames)
            self.assertEqual(len(result["peak_points"]), 13)
            self.assertEqual(result["peak_heap_bytes"], 12049)
            self.assertEqual(result["blocks_at_peak"], 15)
            self.assertEqual(result["end_heap_bytes"], 1)
            self.assertEqual(result["peak_points"][-1]["bytes"], 49)
            log.write_text(log.read_text().replace("At t-end: 1", "At t-end: 2"))
            with self.assertRaisesRegex(ValueError, "totals disagree"):
                summarize_profile(path, log, {})

    def test_grouped_stack_deltas_conserve_bytes_and_blocks(self):
        frames = {
            "retained_history (history.rs:10)": 0,
            "retained_history (history.rs:33)": 1,
            "scratch_buffer (joint_reverse.rs:12)": 2,
            "runtime_overlap (registry.rs:44)": 3,
        }
        baseline = {"peak_heap_bytes": 1000, "blocks_at_peak": 2,
                    "peak_points": [{"bytes": 1000, "blocks": 2, "frames": [0]}]}
        candidate = {"peak_heap_bytes": 6145, "blocks_at_peak": 23,
                     "peak_points": [{"bytes": 600, "blocks": 1, "frames": [1]},
                                     {"bytes": 400, "blocks": 1, "frames": [1]},
                                     {"bytes": 48, "blocks": 3, "frames": [2]},
                                     {"bytes": 5097, "blocks": 18, "frames": [3]}]}
        delta = compare_peak(baseline, candidate, frames)
        self.assertEqual(delta["bytes_delta"], 5145)
        self.assertEqual(delta["blocks_delta"], 21)
        self.assertEqual([g["bytes_delta"] for g in delta["groups"]], [5097, 48])
        reverse = compare_peak(candidate, baseline, frames)
        self.assertEqual(reverse["bytes_delta"], -5145)
        self.assertEqual(reverse["blocks_delta"], -21)


if __name__ == "__main__":
    unittest.main()
