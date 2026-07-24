"""扩大sample评估统计逻辑的CPU单元测试。"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRAMEWORK_ROOT / "src"))

from ptf.sample_eval import paired_prompt_bootstrap, summarize_sample_rows, wilson_interval


def _row(
    prompt_index: int,
    return_index: int,
    *,
    exact: bool,
    format_ok: bool,
    hit_max: bool,
) -> dict[str, object]:
    return {
        "eval_seed": 73001,
        "prompt_index": prompt_index,
        "return_index": return_index,
        "exact_match": exact,
        "format_ok": format_ok,
        "single_final_answer_ok": format_ok,
        "terminated_by_eos": not hit_max,
        "reached_max_tokens_without_eos": hit_max,
        "response_token_count": 100,
        "reward": 1.0 if exact else 0.0,
    }


class SampleStatisticsTest(unittest.TestCase):
    def test_wilson_interval_contains_observed_rate(self) -> None:
        lower, upper = wilson_interval(90, 100)
        self.assertLess(lower, 0.9)
        self.assertGreater(upper, 0.9)
        self.assertAlmostEqual(lower, 0.8256, places=4)
        self.assertAlmostEqual(upper, 0.9448, places=4)

    def test_summary_counts_binary_metrics(self) -> None:
        rows = [
            _row(0, 0, exact=True, format_ok=True, hit_max=False),
            _row(0, 1, exact=False, format_ok=False, hit_max=True),
            _row(1, 0, exact=True, format_ok=True, hit_max=False),
            _row(1, 1, exact=False, format_ok=True, hit_max=False),
        ]
        summary = summarize_sample_rows(rows)
        self.assertEqual(summary["responses"], 4)
        self.assertEqual(summary["prompt_count"], 2)
        self.assertEqual(summary["metrics"]["exact_match"]["successes"], 2)
        self.assertEqual(summary["metrics"]["format_ok"]["rate"], 0.75)
        self.assertEqual(
            summary["metrics"]["reached_max_tokens_without_eos"]["rate"], 0.25
        )

    def test_prompt_bootstrap_is_deterministic_and_clustered(self) -> None:
        control = []
        candidate = []
        for prompt_index in range(10):
            for return_index in range(4):
                control.append(
                    _row(
                        prompt_index,
                        return_index,
                        exact=False,
                        format_ok=return_index < 3,
                        hit_max=return_index == 3,
                    )
                )
                candidate.append(
                    _row(
                        prompt_index,
                        return_index,
                        exact=False,
                        format_ok=True,
                        hit_max=False,
                    )
                )
        first = paired_prompt_bootstrap(
            control, candidate, "format_ok", samples=500, seed=42
        )
        second = paired_prompt_bootstrap(
            control, candidate, "format_ok", samples=500, seed=42
        )
        self.assertEqual(first, second)
        self.assertEqual(first["paired_responses"], 40)
        self.assertEqual(first["prompt_clusters"], 10)
        self.assertEqual(first["delta"], 0.25)
        self.assertEqual(first["ci_low"], 0.25)
        self.assertEqual(first["ci_high"], 0.25)


if __name__ == "__main__":
    unittest.main()

