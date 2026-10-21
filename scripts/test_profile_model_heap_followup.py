"""Guard the heap evidence against omitted tail sites and bad totals."""
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import profile_model_heap_followup as replay
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

    def test_rejects_wrong_replay_targets_and_mismatched_build_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = {label: Path(directory) / label for label in ("baseline", "candidate")}
            sources = {}
            for label, commit in [("baseline", replay.ORIGINAL_BASELINE),
                                  ("candidate", replay.ORIGINAL_CANDIDATE)]:
                roots[label].mkdir()
                (roots[label] / "rust-toolchain.toml").write_text('[toolchain]\nchannel = "1.98.1"\n')
                sources[label] = {"commit": commit, "cargo_lock_sha256": "same-lock"}
            replay.validate_replay_sources(roots, sources)
            for label in roots:
                with self.subTest(wrong_target=label):
                    with patch.dict(sources[label], commit="0" * 40):
                        with self.assertRaisesRegex(ValueError, f"{label} must be the original S7 commit"):
                            replay.validate_replay_sources(roots, sources)
            with patch.dict(sources["candidate"], cargo_lock_sha256="different-lock"):
                with self.assertRaisesRegex(ValueError, "Cargo.lock files must match"):
                    replay.validate_replay_sources(roots, sources)
            (roots["candidate"] / "rust-toolchain.toml").write_text('[toolchain]\nchannel = "other"\n')
            with self.assertRaisesRegex(ValueError, "toolchain files must match"):
                replay.validate_replay_sources(roots, sources)

    def test_reads_retained_original_evidence_from_current_tooling(self):
        with tempfile.TemporaryDirectory() as directory:
            tooling = Path(directory) / "tooling"
            retained = tooling / replay.RETAINED
            retained.parent.mkdir(parents=True)
            original = {"run_id": replay.ORIGINAL_RUN}
            for label, commit, size in [("baseline", replay.ORIGINAL_BASELINE, 10),
                                        ("candidate", replay.ORIGINAL_CANDIDATE, 20)]:
                original[label] = {"source": {"commit": commit}, "checksum": "same-output",
                                   "peak_heap_bytes": size, "blocks_at_peak": 1,
                                   "peak_points": [{"bytes": size, "blocks": 1, "frames": [0]}]}
            previous = {"original": original, "frames": ["allocation (example.rs:1)"]}
            retained.write_text(json.dumps(previous))
            before = retained.read_bytes()
            frames = {}
            with patch.object(replay, "TOOLING_ROOT", tooling):
                result = replay.original_pair(Path(directory) / "absent-artifact", frames)
                self.assertEqual(result["comparison"]["bytes_delta"], 10)
                self.assertEqual(frames, {"allocation (example.rs:1)": 0})
                self.assertEqual(retained.read_bytes(), before)
                original["candidate"]["source"]["commit"] = "0" * 40
                retained.write_text(json.dumps(previous))
                with self.assertRaises(AssertionError):
                    replay.original_pair(Path(directory) / "absent-artifact", {})

    def test_workflow_separates_current_tooling_from_pinned_sources(self):
        workflow = (replay.TOOLING_ROOT / ".github/workflows/model-heap-followup.yml").read_text()
        checkouts = re.findall(
            r"uses: actions/checkout@[^\n]+\n\s+with:\n\s+ref: ([^\n]+)\n\s+path: ([^\n]+)",
            workflow)
        self.assertEqual(len(checkouts), 3)
        self.assertEqual({path: ref for ref, path in checkouts}, {
            "tooling": "${{ github.event.pull_request.head.sha || github.sha }}",
            "baseline": replay.ORIGINAL_BASELINE,
            "candidate": replay.ORIGINAL_CANDIDATE,
        })
        self.assertIn("working-directory: tooling\n        run: python -m unittest discover -s scripts -p test_profile_model_heap_followup.py -v", workflow)
        self.assertIn("working-directory: tooling\n        run: python scripts/profile_model_heap_followup.py ../baseline ../candidate ../heap-followup-results --original-directory ../original-profiles", workflow)
        self.assertIn(f"hashFiles('tooling/{replay.RETAINED.as_posix()}') == ''", workflow)


if __name__ == "__main__":
    unittest.main()
